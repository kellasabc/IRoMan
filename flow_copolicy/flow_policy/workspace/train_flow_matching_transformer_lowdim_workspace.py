import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import wandb
import tqdm
from flow_policy.common.checkpoint_util import TopKCheckpointManager
from flow_policy.common.pytorch_util import dict_apply
from flow_policy.common.replay_buffer import ReplayBuffer
from flow_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from flow_policy.model.common.lr_scheduler import get_scheduler
from flow_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from flow_policy.model.diffusion.flow_matching_transformer import FlowMatchingTransformer
from flow_policy.policy.flow_matching_transformer_lowdim_policy import FlowMatchingTransformerLowdimPolicy
from flow_policy.workspace.base_workspace import BaseWorkspace
from flow_policy.common.json_logger import JsonLogger

logger = logging.getLogger(__name__)

class TrainFlowMatchingTransformerLowdimWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg)
        if output_dir is None:
            output_dir = self.cfg.output_dir
        self._output_dir = output_dir
        device = torch.device(self.cfg.training.device)
        self.device = device
        # debug mode
        if hasattr(self.cfg.training, "debug") and self.cfg.training.debug:
            self.cfg.training.n_epochs = 2
            self.cfg.training.max_train_steps = 3
            self.cfg.training.max_val_steps = 3
            # self.cfg.training.rollout_every = 1  # rollout disabled
            self.cfg.training.checkpoint_every = 1
            self.cfg.training.eval_every = 1
            self.cfg.training.sample_every = 1
        # set up logging
        if self.cfg.training.log:
            wandb.init(
                dir=output_dir,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **self.cfg.logging
            )
        # set up dataset
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        self.normalizer = dataset.get_normalizer()
        train_sampler = SequenceSampler(
            replay_buffer=dataset.replay_buffer,
            sequence_length=self.cfg.horizon,
            pad_before=self.cfg.n_obs_steps - 1,
            pad_after=self.cfg.n_action_steps - 1,
            episode_mask=dataset.train_mask
        )
        dataset.sampler = train_sampler
        # Use dataloader configuration, use defaults if not exists
        dataloader_cfg = getattr(self.cfg, 'dataloader', {})
        train_dataloader_kwargs = {
            'batch_size': dataloader_cfg.get('batch_size', 256),  # Default batch_size
            'shuffle': dataloader_cfg.get('shuffle', False),
            'num_workers': dataloader_cfg.get('num_workers', 0),  # Default single process
            'pin_memory': dataloader_cfg.get('pin_memory', True),
            'persistent_workers': dataloader_cfg.get('persistent_workers', False)
        }
        
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            **train_dataloader_kwargs
        )
        self.model = hydra.utils.instantiate(self.cfg.policy.model)
        self.model.to(device)
        self.policy = hydra.utils.instantiate(
            self.cfg.policy,
            model=self.model,
            device=device
        )
        self.policy.to(device)
        self.policy.mask_generator.to(device)
        self.policy.set_normalizer(self.normalizer)
        self.optimizer = self.policy.get_optimizer(
            weight_decay=self.cfg.training.weight_decay,
            learning_rate=self.cfg.training.lr,
            betas=self.cfg.training.betas
        )
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.training.num_warmup_steps,
            num_training_steps=len(self.train_dataloader) * self.cfg.training.n_epochs,
            name=self.cfg.training.scheduler_type
        )
        val_dataset = dataset.get_validation_dataset()
        # Use val_dataloader configuration, use dataloader config as fallback if not exists
        val_dataloader_cfg = getattr(self.cfg, 'val_dataloader', {})
        dataloader_cfg = getattr(self.cfg, 'dataloader', {})
        val_dataloader_kwargs = {
            'batch_size': val_dataloader_cfg.get('batch_size', dataloader_cfg.get('batch_size', 256)),  # Default batch_size
            'shuffle': val_dataloader_cfg.get('shuffle', False),
            'num_workers': val_dataloader_cfg.get('num_workers', dataloader_cfg.get('num_workers', 0)),  # Default single process
            'pin_memory': val_dataloader_cfg.get('pin_memory', dataloader_cfg.get('pin_memory', True)),
            'persistent_workers': val_dataloader_cfg.get('persistent_workers', dataloader_cfg.get('persistent_workers', False))
        }
        
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            **val_dataloader_kwargs
        )
        self.checkpoint_dir = Path(output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.topk_manager = TopKCheckpointManager(
            save_dir=str(self.checkpoint_dir),
            **self.cfg.checkpoint.topk
        )
        self.train_metrics = {}
        self.val_metrics = {}
        self.global_step = 0
        self.epoch = 0
        self.train_sampling_batch = None
        self.log_path = os.path.join(self._output_dir, "logs.json.txt")

    def run(self):
        logger.info("Starting training...")
        # Create environment runner (even if not using rollout, maintain configuration integrity)
        env_runner = hydra.utils.instantiate(
            self.cfg.task.env_runner, output_dir=self._output_dir
        )
        gradient_accumulate_every = getattr(self.cfg.training, "gradient_accumulate_every", 1)
        with JsonLogger(self.log_path) as json_logger:
            for epoch in range(self.cfg.training.n_epochs):
                step_log = dict()
                train_losses = []
                with tqdm.tqdm(self.train_dataloader, desc=f"Train epoch {epoch}", leave=False, mininterval=getattr(self.cfg.training, 'tqdm_interval_sec', 1.0)) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        try:
                            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                            if self.train_sampling_batch is None:
                                self.train_sampling_batch = batch
                            loss = self.policy.compute_loss(batch)
                            loss = loss / gradient_accumulate_every
                            loss.backward()
                            if (self.global_step % gradient_accumulate_every) == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                self.scheduler.step()
                            train_losses.append(loss.item() * gradient_accumulate_every)
                            tepoch.set_postfix(loss=loss.item() * gradient_accumulate_every, refresh=False)
                            step_log = {
                                "train_loss": loss.item() * gradient_accumulate_every,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                                "lr": self.optimizer.param_groups[0]['lr'],
                            }
                            if hasattr(self.cfg.training, "max_train_steps") and batch_idx >= (self.cfg.training.max_train_steps - 1):
                                break
                            self.global_step += 1
                        except Exception as e:
                            logger.error(f"Train batch error: {e}")
                            import traceback; traceback.print_exc()
                            continue
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss
                
                # ========= rollout (disabled) ==========
                # Comment out rollout functionality, but keep env_runner instantiation
                # if hasattr(self.cfg.training, 'rollout_every') and epoch % self.cfg.training.rollout_every == 0:
                #     try:
                #         self.policy.eval()
                #         rollout_log = env_runner.run(self.policy)
                #         step_log.update(rollout_log)
                #         if self.cfg.training.log:
                #             wandb.log(rollout_log, step=self.global_step)
                #     except Exception as e:
                #         logger.error(f"Rollout error: {e}")
                #         import traceback; traceback.print_exc()
                #     finally:
                #         self.policy.train()
                
                # ========= validation ==========
                if epoch % self.cfg.training.eval_every == 0:
                    try:
                        self.policy.eval()
                        val_losses = []
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(self.val_dataloader):
                                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                                vloss = self.policy.compute_loss(batch)
                                val_losses.append(vloss.item())
                                if hasattr(self.cfg.training, "max_val_steps") and batch_idx >= (self.cfg.training.max_val_steps - 1):
                                    break
                        val_loss = np.mean(val_losses) if val_losses else 0.0
                        step_log["val_loss"] = val_loss
                    except Exception as e:
                        logger.error(f"Validation error: {e}")
                        import traceback; traceback.print_exc()
                    finally:
                        self.policy.train()
                # ========= sample ==========
                if epoch % self.cfg.training.sample_every == 0:
                    try:
                        self.policy.eval()
                        batch = self.train_sampling_batch
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                        with torch.no_grad():
                            if self.cfg.human_act_as_cond:
                                obs_dict = {
                                    "obs": batch["obs"],
                                    "action": batch["action"],
                                    "past_action": batch["past_action"],
                                }
                                gt_action = batch["action"]
                            else:
                                obs_dict = {"obs": batch["obs"], "action": batch["action"]}
                                gt_action = batch["action"]
                            result = self.policy.predict_action(obs_dict)
                            if hasattr(self.cfg, "pred_action_steps_only") and self.cfg.pred_action_steps_only:
                                pred_action = result["action"]
                                start = self.cfg.n_obs_steps - 1
                                end = start + self.cfg.n_action_steps
                                gt_action = gt_action[:, start:end]
                            else:
                                pred_action = result["action_pred"]  # Use complete prediction sequence for MSE calculation
                            mse = F.mse_loss(pred_action, gt_action)
                            step_log["train_action_mse_error"] = mse.item()
                            if self.cfg.training.log:
                                wandb.log({"sample_mse": mse.item(), "epoch": epoch})
                    except Exception as e:
                        logger.error(f"Sample error: {e}")
                        import traceback; traceback.print_exc()
                    finally:
                        self.policy.train()
                # ========= checkpoint ==========
                if epoch % self.cfg.training.checkpoint_every == 0:
                    try:
                        if self.cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint(epoch)
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace("/", "_")
                            metric_dict[new_key] = value
                        topk_ckpt_path = self.topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(epoch, path=topk_ckpt_path)
                    except Exception as e:
                        logger.error(f"Checkpoint error: {e}")
                        import traceback; traceback.print_exc()
                # ========= logging ==========
                if self.cfg.training.log:
                    wandb.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1
        logger.info("Training completed!")

    def save_checkpoint(self, epoch: int, path: Optional[str] = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'normalizer_state_dict': self.normalizer.state_dict(),
            'cfg': self.cfg,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        if path is None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        else:
            checkpoint_path = Path(path)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.normalizer.load_state_dict(checkpoint['normalizer_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'] 
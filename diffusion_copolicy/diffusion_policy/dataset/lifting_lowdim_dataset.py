from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class LiftingLowdimDataset(BaseLowdimDataset):
    """
    Dataset for collaborative lifting task with 23-dimensional observations and 10-dimensional actions.
    
    Observation structure (23 dimensions):
    - robot0_eef_pos: (3D) robot end-effector position
    - robot0_gripper_qpos: (2D) gripper joint positions
    - robot0_gripper_qvel: (2D) gripper joint velocities
    - vec_eef_to_human_head: (3D) vector from end-effector to human head
    - vec_eef_to_human_lh: (3D) vector from end-effector to human left hand
    - vec_eef_to_human_rh: (3D) vector from end-effector to human right hand
    - board_quat: (4D) board orientation quaternion
    - board_balance: (1D) board balance (dot product of board normal and up vector)
    - board_gripped: (1D) whether board is gripped by both fingerpads
    - dist_eef_to_human_head: (1D) distance from end-effector to human head
    
    Action structure (10 dimensions):
    - robot actions (4D): [x_delta, y_delta, z_delta, gripper_action]
    - human actions (6D): [human_left_x, human_left_y, human_left_z, human_right_x, human_right_y, human_right_z]
    """
    
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='obs',
            action_key='action',
            past_action_key='past_action',
            obs_eef_target=True,
            human_act_as_cond=False,
            use_manual_normalizer=False,
            seed=42,
            val_ratio=0.0
            ):
        super().__init__()

        if human_act_as_cond:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=[obs_key, action_key, past_action_key])
        else:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=[obs_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.obs_key = obs_key
        self.action_key = action_key
        self.past_action_key = past_action_key if human_act_as_cond else None
        self.obs_eef_target = obs_eef_target
        self.use_manual_normalizer = use_manual_normalizer
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)

        normalizer = LinearNormalizer()
        if not self.use_manual_normalizer:
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        else:
            # Manual normalizer for lifting task
            x = data['obs']
            stat = {
                'max': np.max(x, axis=0),
                'min': np.min(x, axis=0),
                'mean': np.mean(x, axis=0),
                'std': np.std(x, axis=0)
            }

            # Define masks for different types of observations
            is_x = np.zeros(stat['max'].shape, dtype=bool)
            is_y = np.zeros_like(is_x)
            is_z = np.zeros_like(is_x)
            is_quat = np.zeros_like(is_x)
            is_other = np.zeros_like(is_x)
            
            # robot0_eef_pos (0:3) - x, y, z positions
            is_x[0] = True  # x position
            is_y[1] = True  # y position
            is_z[2] = True  # z position
            
            # robot0_gripper_qpos (3:5) - gripper joint positions
            is_other[3:5] = True
            
            # robot0_gripper_qvel (5:7) - gripper joint velocities
            is_other[5:7] = True
            
            # vec_eef_to_human_head (7:10) - x, y, z vectors
            is_x[7] = True  # x vector
            is_y[8] = True  # y vector
            is_z[9] = True  # z vector
            
            # vec_eef_to_human_lh (10:13) - x, y, z vectors
            is_x[10] = True  # x vector
            is_y[11] = True  # y vector
            is_z[12] = True  # z vector
            
            # vec_eef_to_human_rh (13:16) - x, y, z vectors
            is_x[13] = True  # x vector
            is_y[14] = True  # y vector
            is_z[15] = True  # z vector
            
            # board_quat (16:20) - quaternion
            is_quat[16:20] = True
            
            # board_balance (20) - balance value
            is_other[20] = True
            
            # board_gripped (21) - boolean-like value
            is_other[21] = True
            
            # dist_eef_to_human_head (22) - distance
            is_other[22] = True

            def normalizer_with_masks(stat, masks):
                global_scale = np.ones_like(stat['max'])
                global_offset = np.zeros_like(stat['max'])
                for mask in masks:
                    output_max = 1
                    output_min = -1
                    input_max = stat['max'][mask].max()
                    input_min = stat['min'][mask].min()
                    input_range = input_max - input_min
                    if input_range > 0:
                        scale = (output_max - output_min) / input_range
                        offset = output_min - scale * input_min
                        global_scale[mask] = scale
                        global_offset[mask] = offset
                return SingleFieldLinearNormalizer.create_manual(
                    scale=global_scale,
                    offset=global_offset,
                    input_stats_dict=stat
                )

            normalizer['obs'] = normalizer_with_masks(stat, [is_x, is_y, is_z, is_quat, is_other])
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
                data['action'], last_n_dims=1, mode=mode, **kwargs)
            
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key] # T, D_o
        if not self.obs_eef_target:
            # For lifting task, we might want to zero out certain observations
            # This is task-specific and can be adjusted
            pass
            
        if self.past_action_key is not None:
            data = {
                'obs': obs,
                'action': sample[self.action_key], # T, D_a
                'past_action': sample[self.past_action_key], # T, D_a
            }
        else:
            data = {
                'obs': obs,
                'action': sample[self.action_key], # T, D_a
            }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

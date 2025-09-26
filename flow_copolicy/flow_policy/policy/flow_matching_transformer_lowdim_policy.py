from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from flow_policy.model.common.normalizer import LinearNormalizer
from flow_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from flow_policy.model.diffusion.flow_matching_transformer import FlowMatchingTransformer
from flow_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class FlowMatchingTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: FlowMatchingTransformer,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            human_act_as_cond=False,
            pred_action_steps_only=False,
            robot_action_dim=2,  # Robot arm action dimension
            human_action_dim=2,  # Human action dimension (carrying table task)
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.human_act_as_cond = human_act_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.robot_action_dim = robot_action_dim
        self.human_action_dim = human_action_dim
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = 50  # Flow Matching typically requires fewer steps
        self.num_inference_steps = num_inference_steps
        self.inference_temperature = 1.0  # Add temperature parameter for noise control during inference
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model

        # Flow Matching sampling
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # Set timesteps
        timesteps = torch.linspace(0, 1, self.num_inference_steps, device=trajectory.device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output (velocity field)
            model_output = model(trajectory, t, cond)

            # 3. update trajectory using Euler method
            dt = t_next - t
            trajectory = trajectory + dt * model_output
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        obs = nobs['obs']
        
        # Use model's device instead of cfg
        device = next(self.model.parameters()).device
        obs = obs.to(device)
        
        # handle different ways of passing observation
        cond = None
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.human_act_as_cond:
                past_action = nobs['past_action']
                past_action = past_action.to(device)
                # Use configured robot_action_dim to get human action dimension
                # past_action structure: [robot_action(0:robot_action_dim), human_action(robot_action_dim:)]
                # So human_action starts from robot_action_dim
                cond_human = past_action[:,:self.n_obs_steps, self.robot_action_dim:] # human previous actions as conditioning
                cond = torch.cat([cond, cond_human], dim=-1)
        
        # sample
        B = obs.shape[0]
        device = obs.device
        
        # Directly use model for inference
        with torch.no_grad():
            # Create initial noise
            if self.pred_action_steps_only:
                sample = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
            else:
                sample = torch.randn((B, self.horizon, self.action_dim), device=device)
            
            # Use Flow Matching for inference, improved numerical stability
            timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)
            for i in range(self.num_inference_steps):
                t = timesteps[i].expand(B)
                t_next = timesteps[i + 1].expand(B)
                
                # Predict velocity field
                pred_velocity = self.model(sample, t, cond)
                
                # Use more stable integration method
                dt = t_next - t
                sample = sample - dt.unsqueeze(-1).unsqueeze(-1) * pred_velocity
            
            # unnormalize
            action = self.normalizer['action'].unnormalize(sample)
            
            # Ensure action is on correct device
            action = action.to(device)
        
        # get action - Keep consistent with diffusion_copolicy
        if self.pred_action_steps_only:
            action_result = action
        else:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action_result = action[:, start:end]
        
        result = {
            'action': action_result,
            'action_pred': action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        past_action = nbatch['past_action'] if self.human_act_as_cond else None

        # Ensure all tensors are on correct device
        device = next(self.model.parameters()).device
        obs = obs.to(device)
        action = action.to(device)
        if past_action is not None:
            past_action = past_action.to(device)

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond: # true default
            cond = obs[:,:self.n_obs_steps,:]
            if self.human_act_as_cond:
                # Use configured robot_action_dim to get human action dimension
                # past_action structure: [robot_action(0:robot_action_dim), human_action(robot_action_dim:)]
                # So human_action starts from robot_action_dim
                cond_human = past_action[:,:self.n_obs_steps, self.robot_action_dim:] # human previous actions as conditioning
                cond = torch.cat([cond, cond_human], dim=-1)
            
            # Keep condition tensor's original shape (B, To, cond_dim)
            # No need to reshape, model expects 3D tensor
            if self.pred_action_steps_only: # false default - only pred # of  action steps not full T
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            # Directly use trajectory's device to create mask
            device = trajectory.device
            condition_mask = self.mask_generator(trajectory.shape).to(device)

        # Flow Matching loss
        B, T, D = trajectory.shape
        
        # Sample random time
        t = torch.rand(B, device=trajectory.device)
        
        # Sample noise
        x_0 = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Create noisy trajectory
        current_trajectory =(1- t.unsqueeze(-1).unsqueeze(-1)) * trajectory + t.unsqueeze(-1).unsqueeze(-1) * x_0
        
        # apply conditioning
        current_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict velocity field
        pred_velocity = self.model(current_trajectory, t, cond)
        
        # Target velocity is the noise
        target_velocity = trajectory - x_0
        
        # compute loss mask
        loss_mask = ~condition_mask

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss 
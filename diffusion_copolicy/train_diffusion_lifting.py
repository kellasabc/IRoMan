#!/usr/bin/env python3
"""
Diffusion Transformer Training Script - Collaborative Lifting Task
Using collaborative_lifting_sac_350.zarr training data
- Observation state: 23-dimensional
- Action state: 10-dimensional (4-dimensional robot action + 6-dimensional human action)
- Using diffusion co-policy model prediction and inference formulas
- Main differences from Flow Matching:
  1. Prediction target: Diffusion predicts noise (epsilon), Flow Matching predicts velocity field
  2. Loss function: Diffusion uses MSE loss to predict noise, Flow Matching uses MSE loss to predict velocity field
  3. Inference process: Diffusion uses DDPM sampler for step-by-step denoising, Flow Matching uses ODE solver
"""

import os
import sys
import hydra
from omegaconf import OmegaConf
import torch

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@hydra.main(
    version_base=None, 
    config_path="diffusion_policy/config", 
    config_name="train_diffusion_transformer_lowdim_lifting"
)
def main(cfg):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create training workspace
    from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import TrainDiffusionTransformerLowdimWorkspace
    
    workspace = TrainDiffusionTransformerLowdimWorkspace(cfg=cfg)
    
    # Start training
    print("Starting Diffusion Transformer training - Collaborative Lifting Task...")
    print(f"Observation dimension: {cfg.obs_dim}")
    print(f"Action dimension: {cfg.action_dim}")
    print(f"Human action as condition: {cfg.human_act_as_cond}")
    print(f"Robot action dimension: {cfg.robot_action_dim}")
    print(f"Human action dimension: {cfg.human_action_dim}")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    print(f"Output directory will be automatically managed by Hydra")
    
    print("\n=== Main Differences between Diffusion vs Flow Matching ===")
    print("1. Prediction target:")
    print("   - Diffusion: Predict noise (epsilon)")
    print("   - Flow Matching: Predict velocity field")
    print("2. Loss function:")
    print("   - Diffusion: MSE(predicted noise, true noise)")
    print("   - Flow Matching: MSE(predicted velocity field, true velocity field)")
    print("3. Inference process:")
    print("   - Diffusion: DDPM sampler step-by-step denoising")
    print("   - Flow Matching: ODE solver integration")
    print("4. Network structure: Same (Transformer)")
    print("==========================================\n")
    
    try:
        workspace.run()
        print("Training completed!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        raise

if __name__ == "__main__":
    main()

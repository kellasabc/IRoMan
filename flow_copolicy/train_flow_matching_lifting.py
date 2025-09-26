#!/usr/bin/env python3
"""
Flow Matching Transformer Training Script
Using collaborative_lifting_sac_350.zarr training data
- Observation state: 23-dimensional
- Action state: 10-dimensional
- Not using rollout functionality
"""

import os
import sys
import hydra
from omegaconf import OmegaConf
import torch

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@hydra.main(version_base=None, config_path="flow_policy/config", config_name="train_flow_matching_collaborative_lifting")
def main(cfg):
    # Set output directory
    output_dir = "outputs/flow_matching_collaborative_lifting"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create training workspace
    from flow_policy.workspace.train_flow_matching_transformer_lowdim_workspace import TrainFlowMatchingTransformerLowdimWorkspace
    
    workspace = TrainFlowMatchingTransformerLowdimWorkspace(
        cfg=cfg,
        output_dir=output_dir
    )
    
    # Start training
    print("Starting Flow Matching Transformer training...")
    print(f"Observation dimension: {cfg.obs_dim}")
    print(f"Action dimension: {cfg.action_dim}")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    print(f"Output directory: {output_dir}")
    
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
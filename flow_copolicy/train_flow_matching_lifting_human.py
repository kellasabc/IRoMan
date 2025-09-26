#!/usr/bin/env python3
"""
Flow Matching Transformer Training Script - Human Action Condition for Collaborative Lifting
Using collaborative_lifting_sac_350.zarr training data
- Observation state: 23-dimensional
- Action state: 10-dimensional (4-dimensional robot action + 6-dimensional human action)
- Using human action as condition
"""

import os
import sys
import logging
import torch
import hydra
from omegaconf import OmegaConf

# Add flow_policy to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flow_policy'))

from flow_policy.workspace.train_flow_matching_transformer_lowdim_workspace import TrainFlowMatchingTransformerLowdimWorkspace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config_path = "flow_policy/config/train_flow_matching_collaborative_lifting_human_cond.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file does not exist: {config_path}")
        return
    
    # Use hydra to load configuration
    config_dir = os.path.abspath("flow_policy/config")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name="train_flow_matching_collaborative_lifting_human_cond")
    
    logger.info("Starting Flow Matching Transformer training (Human Action Condition for Collaborative Lifting)...")
    logger.info(f"Observation dimension: {cfg.obs_dim}")
    logger.info(f"Action dimension: {cfg.action_dim}")
    logger.info(f"Human Action as Condition: {cfg.human_act_as_cond}")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Set output directory
    output_dir = "outputs/flow_matching_collaborative_lifting_human_cond"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Create training workspace
        workspace = TrainFlowMatchingTransformerLowdimWorkspace(cfg, output_dir=output_dir)
        
        # Start training
        workspace.run()
        
        logger.info("Training completed!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Flow Matching Transformer Training Script - Table Carrying Task (No Human Action Condition)
Ensure not using human action as condition
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
    config_path = "flow_policy/config/train_flow_matching_transformer_lowdim_table_workspace.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file does not exist: {config_path}")
        return
    
    # Use hydra to load configuration
    config_dir = os.path.abspath("flow_policy/config")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name="train_flow_matching_transformer_lowdim_table_workspace")
    
    # Ensure not using human action as condition
    cfg.human_act_as_cond = False
    cfg.policy.human_act_as_cond = False
    cfg.policy.model.human_act_as_cond = False
    cfg.task.dataset.human_act_as_cond = False
    cfg.task.env_runner.past_action = False
    # Set dataset path
    cfg.task.dataset.zarr_path = "data/table/table_10Hz.zarr"
    
    logger.info("Starting Flow Matching Transformer training (Table Carrying Task - No Human Condition)...")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"human_act_as_cond setting: {cfg.human_act_as_cond}")
    
    # Set output directory
    output_dir = "outputs/flow_matching_table_no_human_cond"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Create training workspace
        workspace = TrainFlowMatchingTransformerLowdimWorkspace(cfg, output_dir=output_dir)
        
        # Start training
        workspace.run()
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

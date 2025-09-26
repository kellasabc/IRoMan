#!/usr/bin/env python3
"""
调试模型加载过程
"""

import torch
import dill
import os
import sys
from omegaconf import OmegaConf
import hydra

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.join(current_dir, '..', '..')
sys.path.append(os.path.join(workspace_root, 'flow_copolicy'))
sys.path.append(os.path.join(workspace_root, 'flow_copolicy', 'flow_policy'))

def debug_model_loading():
    model_path = "table-carrying-ai/trained_models/flowmatching/model_10Hz.ckpt"
    
    print("=== 调试模型加载过程 ===")
    
    # 1. 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', pickle_module=dill)
    cfg = checkpoint['cfg']
    
    print("1. 原始配置:")
    print(f"   human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
    print(f"   cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    
    # 2. 模拟设置 human_act_as_cond = False
    print("\n2. 设置 human_act_as_cond = False:")
    OmegaConf.update(cfg, "human_act_as_cond", False, merge=False)
    OmegaConf.update(cfg, "task.dataset.human_act_as_cond", False, merge=False)
    OmegaConf.update(cfg, "policy.human_act_as_cond", False, merge=False)
    
    print(f"   human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
    print(f"   cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    
    # 3. 检查是否需要更新 cond_dim
    print("\n3. 检查 cond_dim 是否需要更新:")
    obs_dim = cfg.get('obs_dim', 18)
    human_act_as_cond = cfg.get('human_act_as_cond', False)
    
    if human_act_as_cond:
        expected_cond_dim = obs_dim + 2  # obs_dim + human action dim
    else:
        expected_cond_dim = obs_dim  # only obs_dim
    
    current_cond_dim = cfg.policy.model.get('cond_dim', 0)
    print(f"   当前 cond_dim: {current_cond_dim}")
    print(f"   期望 cond_dim: {expected_cond_dim}")
    print(f"   是否需要更新: {current_cond_dim != expected_cond_dim}")
    
    # 4. 更新 cond_dim
    if current_cond_dim != expected_cond_dim:
        print("\n4. 更新 cond_dim:")
        OmegaConf.update(cfg.policy.model, "cond_dim", expected_cond_dim, merge=False)
        print(f"   更新后 cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    
    # 5. 尝试实例化模型
    print("\n5. 尝试实例化模型:")
    try:
        from flow_policy.model.diffusion.flow_matching_transformer import FlowMatchingTransformer
        model = hydra.utils.instantiate(cfg.policy.model)
        print("   ✓ 模型实例化成功")
        
        # 检查模型的 cond_obs_emb 层
        if hasattr(model, 'cond_obs_emb'):
            print(f"   cond_obs_emb 输入维度: {model.cond_obs_emb.in_features}")
            print(f"   cond_obs_emb 输出维度: {model.cond_obs_emb.out_features}")
        
    except Exception as e:
        print(f"   ✗ 模型实例化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_loading()

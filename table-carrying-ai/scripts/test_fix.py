#!/usr/bin/env python3
"""
测试修复是否有效
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

def test_fix():
    model_path = "table-carrying-ai/trained_models/flowmatching/model_10Hz.ckpt"
    
    print("=== 测试修复是否有效 ===")
    
    # 1. 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', pickle_module=dill)
    cfg = checkpoint['cfg']
    
    print("1. 原始配置:")
    print(f"   human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
    print(f"   cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    
    # 2. 模拟修复逻辑
    print("\n2. 应用修复逻辑:")
    
    # 设置 human_act_as_cond = False
    human_act_as_cond = False
    OmegaConf.update(cfg, "human_act_as_cond", human_act_as_cond, merge=False)
    OmegaConf.update(cfg, "task.dataset.human_act_as_cond", human_act_as_cond, merge=False)
    OmegaConf.update(cfg, "policy.human_act_as_cond", human_act_as_cond, merge=False)
    print(f"   ✓ 设置 human_act_as_cond = {human_act_as_cond}")
    
    # 确保 cond_dim 与 human_act_as_cond 设置一致
    obs_dim = cfg.get('obs_dim', 18)
    if human_act_as_cond:
        expected_cond_dim = obs_dim + 2  # obs_dim + human action dim
    else:
        expected_cond_dim = obs_dim  # only obs_dim
    
    current_cond_dim = cfg.policy.model.get('cond_dim', 0)
    if current_cond_dim != expected_cond_dim:
        print(f"   ✓ 更新 cond_dim: {current_cond_dim} -> {expected_cond_dim}")
        OmegaConf.update(cfg.policy.model, "cond_dim", expected_cond_dim, merge=False)
    else:
        print(f"   ✓ cond_dim 已正确: {current_cond_dim}")
    
    print(f"   最终配置:")
    print(f"   human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
    print(f"   cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    
    # 3. 测试模型实例化和前向传播
    print("\n3. 测试模型实例化和前向传播:")
    try:
        from flow_policy.model.diffusion.flow_matching_transformer import FlowMatchingTransformer
        model = hydra.utils.instantiate(cfg.policy.model)
        print("   ✓ 模型实例化成功")
        
        # 测试前向传播
        batch_size = 1
        horizon = cfg.get('horizon', 12)
        action_dim = cfg.get('action_dim', 4)
        cond_dim = cfg.policy.model.get('cond_dim', 18)
        
        # 创建测试输入
        sample = torch.randn(batch_size, horizon, action_dim)
        t = torch.ones(batch_size)
        cond = torch.randn(batch_size, 3, cond_dim)  # 3个观察步
        
        print(f"   测试输入:")
        print(f"   sample shape: {sample.shape}")
        print(f"   t shape: {t.shape}")
        print(f"   cond shape: {cond.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(sample, t, cond)
            print(f"   ✓ 前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fix()









#!/usr/bin/env python3
"""
深度检查配置问题
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

def deep_check_config():
    model_path = "table-carrying-ai/trained_models/flowmatching/model_10Hz.ckpt"
    
    print("=== 深度检查配置问题 ===")
    
    # 1. 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', pickle_module=dill)
    cfg = checkpoint['cfg']
    
    print("1. Checkpoint 中的配置:")
    print(f"   human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
    print(f"   cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    print(f"   obs_dim: {cfg.get('obs_dim', 'Not found')}")
    print(f"   obs_as_cond: {cfg.get('obs_as_cond', 'Not found')}")
    
    # 2. 检查模型状态字典
    print("\n2. 模型状态字典分析:")
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    elif "state_dicts" in checkpoint:
        model_state_dict = checkpoint["state_dicts"]["model"]
    else:
        print("   未找到模型状态字典")
        return
    
    # 检查关键层的维度
    key_layers = ["cond_obs_emb.weight", "input_emb.weight", "pos_emb"]
    for key in key_layers:
        if key in model_state_dict:
            shape = model_state_dict[key].shape
            print(f"   {key}: {shape}")
        else:
            print(f"   {key}: 未找到")
    
    # 3. 模拟配置解析过程
    print("\n3. 模拟配置解析过程:")
    
    # 检查是否有 eval 表达式
    cond_dim_config = cfg.policy.model.get('cond_dim', None)
    print(f"   原始 cond_dim 配置: {cond_dim_config}")
    print(f"   配置类型: {type(cond_dim_config)}")
    
    # 检查是否应该动态计算
    obs_dim = cfg.get('obs_dim', 18)
    obs_as_cond = cfg.get('obs_as_cond', True)
    human_act_as_cond = cfg.get('human_act_as_cond', False)
    
    print(f"   obs_dim: {obs_dim}")
    print(f"   obs_as_cond: {obs_as_cond}")
    print(f"   human_act_as_cond: {human_act_as_cond}")
    
    # 计算期望的 cond_dim
    if obs_as_cond and human_act_as_cond:
        expected_cond_dim = obs_dim + 2
    elif obs_as_cond:
        expected_cond_dim = obs_dim
    else:
        expected_cond_dim = 0
    
    print(f"   期望的 cond_dim: {expected_cond_dim}")
    print(f"   实际的 cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    
    # 4. 检查是否有配置覆盖
    print("\n4. 检查配置覆盖:")
    
    # 检查所有可能的配置来源
    config_sources = [
        ("根配置", cfg),
        ("policy配置", cfg.get('policy', {})),
        ("model配置", cfg.get('policy', {}).get('model', {}))
    ]
    
    for name, config in config_sources:
        if 'human_act_as_cond' in config:
            print(f"   {name} human_act_as_cond: {config['human_act_as_cond']}")
        if 'cond_dim' in config:
            print(f"   {name} cond_dim: {config['cond_dim']}")

if __name__ == "__main__":
    deep_check_config()









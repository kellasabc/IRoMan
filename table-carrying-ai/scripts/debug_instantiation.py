#!/usr/bin/env python3
"""
调试模型实例化过程
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

def debug_instantiation():
    model_path = "table-carrying-ai/trained_models/flowmatching/model_10Hz.ckpt"
    
    print("=== 调试模型实例化过程 ===")
    
    # 1. 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', pickle_module=dill)
    cfg = checkpoint['cfg']
    
    print("1. 配置信息:")
    print(f"   human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
    print(f"   cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
    print(f"   obs_dim: {cfg.get('obs_dim', 'Not found')}")
    
    # 2. 获取模型状态字典
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    elif "state_dicts" in checkpoint:
        model_state_dict = checkpoint["state_dicts"]["model"]
    else:
        print("未找到模型状态字典")
        return
    
    print(f"\n2. 模型状态字典中的 cond_obs_emb.weight 维度:")
    if "cond_obs_emb.weight" in model_state_dict:
        shape = model_state_dict["cond_obs_emb.weight"].shape
        print(f"   cond_obs_emb.weight: {shape}")
        actual_cond_dim = shape[1]
    else:
        print("   未找到 cond_obs_emb.weight")
        return
    
    # 3. 检查配置是否匹配
    config_cond_dim = cfg.policy.model.get('cond_dim', 0)
    print(f"\n3. 配置匹配检查:")
    print(f"   配置中的 cond_dim: {config_cond_dim}")
    print(f"   状态字典中的 cond_dim: {actual_cond_dim}")
    print(f"   是否匹配: {config_cond_dim == actual_cond_dim}")
    
    if config_cond_dim != actual_cond_dim:
        print(f"   ⚠ 配置不匹配！需要修正配置")
        
        # 修正配置
        print(f"\n4. 修正配置:")
        print(f"   修正前 cond_dim: {config_cond_dim}")
        OmegaConf.update(cfg.policy.model, "cond_dim", actual_cond_dim, merge=False)
        print(f"   修正后 cond_dim: {cfg.policy.model.get('cond_dim')}")
        
        # 根据实际维度推断 human_act_as_cond
        obs_dim = cfg.get('obs_dim', 18)
        if actual_cond_dim == obs_dim + 2:
            inferred_human_act_as_cond = True
            print(f"   推断 human_act_as_cond: True (cond_dim = {obs_dim} + 2)")
        elif actual_cond_dim == obs_dim:
            inferred_human_act_as_cond = False
            print(f"   推断 human_act_as_cond: False (cond_dim = {obs_dim})")
        else:
            print(f"   ⚠ 未知的 cond_dim: {actual_cond_dim}")
            inferred_human_act_as_cond = False
        
        # 更新配置
        OmegaConf.update(cfg, "human_act_as_cond", inferred_human_act_as_cond, merge=False)
        OmegaConf.update(cfg, "task.dataset.human_act_as_cond", inferred_human_act_as_cond, merge=False)
        OmegaConf.update(cfg, "policy.human_act_as_cond", inferred_human_act_as_cond, merge=False)
    
    # 5. 尝试实例化模型
    print(f"\n5. 尝试实例化模型:")
    try:
        from flow_policy.model.diffusion.flow_matching_transformer import FlowMatchingTransformer
        model = hydra.utils.instantiate(cfg.policy.model)
        print("   ✓ 模型实例化成功")
        
        # 检查模型的 cond_obs_emb 层
        if hasattr(model, 'cond_obs_emb') and model.cond_obs_emb is not None:
            print(f"   cond_obs_emb 输入维度: {model.cond_obs_emb.in_features}")
            print(f"   cond_obs_emb 输出维度: {model.cond_obs_emb.out_features}")
        
        # 尝试加载状态字典
        model.load_state_dict(model_state_dict)
        print("   ✓ 状态字典加载成功")
        
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_instantiation()









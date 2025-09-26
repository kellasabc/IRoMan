#!/usr/bin/env python3
"""
检查模型配置
"""

import torch
import dill
import os

def check_model_config():
    # 检查两个模型
    models = [
        ("model_10Hz.ckpt", "table-carrying-ai/trained_models/flowmatching/model_10Hz.ckpt"),
        ("model_human_act_as_cond_10Hz.ckpt", "table-carrying-ai/trained_models/flowmatching/model_human_act_as_cond_10Hz.ckpt")
    ]
    
    for model_name, model_path in models:
        print(f"\n{'='*50}")
        print(f"检查模型: {model_name}")
        print(f"{'='*50}")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            continue
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', pickle_module=dill)
            cfg = checkpoint['cfg']
            
            print("配置信息:")
            print(f"human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
            print(f"cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
            print(f"obs_dim: {cfg.get('obs_dim', 'Not found')}")
            print(f"action_dim: {cfg.get('action_dim', 'Not found')}")
            print(f"n_obs_steps: {cfg.get('n_obs_steps', 'Not found')}")
            print(f"n_action_steps: {cfg.get('n_action_steps', 'Not found')}")
            
            # 检查是否使用人类条件
            human_act_as_cond = cfg.get('human_act_as_cond', False)
            cond_dim = cfg.policy.model.get('cond_dim', 0)
            obs_dim = cfg.get('obs_dim', 0)
            
            print(f"\n分析结果:")
            if human_act_as_cond:
                print("✓ 该模型使用人类action作为条件")
            else:
                print("✗ 该模型不使用人类action作为条件")
                
            if cond_dim == obs_dim:
                print("✓ 条件维度等于观察维度，符合不使用人类条件的配置")
            elif cond_dim == obs_dim + 2:
                print("✓ 条件维度等于观察维度+2，符合使用人类条件的配置")
            else:
                print(f"⚠ 条件维度异常: {cond_dim} (观察维度: {obs_dim})")
                
        except Exception as e:
            print(f"读取模型配置时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', pickle_module=dill)
        cfg = checkpoint['cfg']
        
        print("=== model_10Hz.ckpt 配置信息 ===")
        print(f"human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
        print(f"cond_dim: {cfg.policy.model.get('cond_dim', 'Not found')}")
        print(f"obs_dim: {cfg.get('obs_dim', 'Not found')}")
        print(f"action_dim: {cfg.get('action_dim', 'Not found')}")
        print(f"n_obs_steps: {cfg.get('n_obs_steps', 'Not found')}")
        print(f"n_action_steps: {cfg.get('n_action_steps', 'Not found')}")
        
        # 检查是否使用人类条件
        human_act_as_cond = cfg.get('human_act_as_cond', False)
        cond_dim = cfg.policy.model.get('cond_dim', 0)
        obs_dim = cfg.get('obs_dim', 0)
        
        print(f"\n=== 分析结果 ===")
        if human_act_as_cond:
            print("✓ 该模型使用人类action作为条件")
        else:
            print("✗ 该模型不使用人类action作为条件")
            
        if cond_dim == obs_dim:
            print("✓ 条件维度等于观察维度，符合不使用人类条件的配置")
        elif cond_dim == obs_dim + 2:
            print("✓ 条件维度等于观察维度+2，符合使用人类条件的配置")
        else:
            print(f"⚠ 条件维度异常: {cond_dim} (观察维度: {obs_dim})")
            
    except Exception as e:
        print(f"读取模型配置时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_config()

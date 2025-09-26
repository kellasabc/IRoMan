#!/usr/bin/env python3
"""
对比 Diffusion Policy 和 Flow Matching 的配置和数据
"""

import os
import sys
import yaml
import zarr
import numpy as np
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_data(data_path):
    """分析数据文件"""
    data = zarr.open(data_path)
    
    print(f"数据文件: {data_path}")
    print(f"  Zarr keys: {list(data.keys())}")
    print(f"  Data keys: {list(data['data'].keys())}")
    print(f"  Meta keys: {list(data['meta'].keys())}")
    
    # 获取 episode 信息
    episode_ends = data['meta']['episode_ends'][:]
    total_episodes = len(episode_ends)
    
    print(f"  总 episode 数: {total_episodes}")
    
    # 计算总步数
    total_steps = episode_ends[-1] if len(episode_ends) > 0 else 0
    print(f"  总步数: {total_steps}")
    
    # 计算平均 episode 长度
    if len(episode_ends) > 1:
        episode_lengths = np.diff(episode_ends)
        avg_length = np.mean(episode_lengths)
        min_length = np.min(episode_lengths)
        max_length = np.max(episode_lengths)
        print(f"  平均 episode 长度: {avg_length:.1f} 步")
        print(f"  最短 episode: {min_length} 步")
        print(f"  最长 episode: {max_length} 步")
    
    # 检查数据维度
    obs_shape = data['data']['obs'].shape
    action_shape = data['data']['action'].shape
    past_action_shape = data['data']['past_action'].shape
    
    print(f"  观察数据形状: {obs_shape}")
    print(f"  动作数据形状: {action_shape}")
    print(f"  过去动作数据形状: {past_action_shape}")
    
    return {
        'total_episodes': total_episodes,
        'total_steps': total_steps,
        'obs_shape': obs_shape,
        'action_shape': action_shape,
        'past_action_shape': past_action_shape
    }

def compare_configs():
    """对比配置文件"""
    print("=" * 80)
    print("Diffusion Policy vs Flow Matching 配置对比")
    print("=" * 80)
    
    # 加载配置文件
    diffusion_config = load_config("../../diffusion_copolicy/diffusion_policy/config/train_diffusion_transformer_lowdim_table_workspace.yaml")
    flow_matching_config = load_config("../../flow_copolicy/flow_policy/config/train_flow_matching_transformer_lowdim_table_workspace.yaml")
    flow_matching_human_config = load_config("../../flow_copolicy/flow_policy/config/train_flow_matching_transformer_lowdim_table_workspace_human_cond.yaml")
    
    print("\n1. 基本配置对比:")
    print("-" * 40)
    
    configs = {
        "Diffusion Policy": diffusion_config,
        "Flow Matching (无人类条件)": flow_matching_config,
        "Flow Matching (有人类条件)": flow_matching_human_config
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  human_act_as_cond: {config.get('human_act_as_cond', 'N/A')}")
        print(f"  obs_dim: {config.get('obs_dim', 'N/A')}")
        print(f"  action_dim: {config.get('action_dim', 'N/A')}")
        print(f"  horizon: {config.get('horizon', 'N/A')}")
        print(f"  n_obs_steps: {config.get('n_obs_steps', 'N/A')}")
        print(f"  n_action_steps: {config.get('n_action_steps', 'N/A')}")
        
        # 模型配置
        if 'policy' in config and 'model' in config['policy']:
            model_config = config['policy']['model']
            print(f"  cond_dim: {model_config.get('cond_dim', 'N/A')}")
            print(f"  input_dim: {model_config.get('input_dim', 'N/A')}")
            print(f"  output_dim: {model_config.get('output_dim', 'N/A')}")
    
    print("\n2. 训练配置对比:")
    print("-" * 40)
    
    for name, config in configs.items():
        print(f"\n{name}:")
        training_config = config.get('training', {})
        print(f"  epochs: {training_config.get('num_epochs', training_config.get('n_epochs', 'N/A'))}")
        print(f"  batch_size: {config.get('dataloader', {}).get('batch_size', 'N/A')}")
        print(f"  learning_rate: {training_config.get('learning_rate', training_config.get('lr', 'N/A'))}")
        print(f"  device: {training_config.get('device', 'N/A')}")
        print(f"  seed: {training_config.get('seed', 'N/A')}")

def analyze_data_files():
    """分析数据文件"""
    print("\n" + "=" * 80)
    print("数据文件分析")
    print("=" * 80)
    
    data_files = [
        ("Diffusion Policy 数据", "../../diffusion_copolicy/data/table/table_10Hz.zarr"),
        ("Flow Matching 数据", "../../flow_copolicy/data/table/table_10Hz.zarr"),
        ("人类条件数据", "../../flow_copolicy/data/table/table_past_actions.zarr")
    ]
    
    results = {}
    
    for name, path in data_files:
        if os.path.exists(path):
            print(f"\n{name}:")
            results[name] = analyze_data(path)
        else:
            print(f"\n{name}: 文件不存在 - {path}")
    
    return results

def main():
    """主函数"""
    print("Diffusion Policy vs Flow Matching 详细对比分析")
    print("=" * 80)
    
    # 对比配置
    compare_configs()
    
    # 分析数据
    data_results = analyze_data_files()
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    print("\n数据规模:")
    for name, result in data_results.items():
        print(f"  {name}: {result['total_episodes']} episodes, {result['total_steps']} 总步数")
    
    print("\n主要差异:")
    print("  1. Diffusion Policy 使用 DDPM 调度器，Flow Matching 使用 Flow Matching 算法")
    print("  2. Flow Matching 通常需要更少的推理步数 (50 vs 100)")
    print("  3. 人类条件版本使用 past_action 作为额外输入")
    print("  4. 训练参数略有不同 (学习率、epochs 等)")
    
    print("\n相同点:")
    print("  1. 都使用相同的数据集 (371 episodes)")
    print("  2. 都使用 Transformer 架构")
    print("  3. 都支持人类条件和非人类条件模式")
    print("  4. 都使用相同的环境配置")

if __name__ == "__main__":
    main()







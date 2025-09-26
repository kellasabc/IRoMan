#!/usr/bin/env python3
"""
检查训练日志，分析配置问题
"""

import os
import json
import glob

def check_training_logs():
    # 检查 flow_copolicy 的输出目录
    flow_output_dir = "flow_copolicy/outputs"
    
    print("=== 检查 Flow Matching 训练日志 ===")
    
    if not os.path.exists(flow_output_dir):
        print(f"输出目录不存在: {flow_output_dir}")
        return
    
    # 查找所有输出目录
    output_dirs = glob.glob(os.path.join(flow_output_dir, "*"))
    
    for output_dir in output_dirs:
        if os.path.isdir(output_dir):
            print(f"\n检查目录: {output_dir}")
            
            # 检查日志文件
            log_file = os.path.join(output_dir, "logs.json.txt")
            if os.path.exists(log_file):
                print(f"  找到日志文件: {log_file}")
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # 读取第一行（通常是配置信息）
                            first_line = json.loads(lines[0])
                            if 'cfg' in first_line:
                                cfg = first_line['cfg']
                                print(f"  训练配置:")
                                print(f"    human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
                                print(f"    cond_dim: {cfg.get('policy', {}).get('model', {}).get('cond_dim', 'Not found')}")
                                print(f"    n_action_steps: {cfg.get('n_action_steps', 'Not found')}")
                except Exception as e:
                    print(f"  读取日志文件失败: {e}")
            
            # 检查 checkpoint 文件
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
                if checkpoint_files:
                    print(f"  找到 {len(checkpoint_files)} 个 checkpoint 文件")
                    # 检查最新的 checkpoint
                    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                    print(f"  最新 checkpoint: {os.path.basename(latest_checkpoint)}")
                    
                    try:
                        import torch
                        import dill
                        checkpoint = torch.load(latest_checkpoint, map_location='cpu', pickle_module=dill)
                        if 'cfg' in checkpoint:
                            cfg = checkpoint['cfg']
                            print(f"  Checkpoint 配置:")
                            print(f"    human_act_as_cond: {cfg.get('human_act_as_cond', 'Not found')}")
                            print(f"    cond_dim: {cfg.get('policy', {}).get('model', {}).get('cond_dim', 'Not found')}")
                            print(f"    n_action_steps: {cfg.get('n_action_steps', 'Not found')}")
                    except Exception as e:
                        print(f"  读取 checkpoint 失败: {e}")

if __name__ == "__main__":
    check_training_logs()









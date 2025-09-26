#!/usr/bin/env python3
"""
测试参数检测逻辑
"""

import os
import sys

# 模拟参数
class MockArgs:
    def __init__(self):
        self.run_mode = "hil"
        self.robot_mode = "planner"
        self.human_mode = "real"
        self.human_control = "joystick"
        self.render_mode = "gui"
        self.planner_type = "flowmatching"
        self.map_config = "cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml"
        self.artifact_path = None
        self.human_act_as_cond = None

def test_parameter_detection():
    # 获取工作空间根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.join(current_dir, '..', '..')
    
    # 创建模拟参数
    exp_args = MockArgs()
    
    print("测试参数检测逻辑:")
    print(f"run_mode: {exp_args.run_mode}")
    print(f"robot_mode: {exp_args.robot_mode}")
    print(f"human_mode: {exp_args.human_mode}")
    print(f"human_control: {exp_args.human_control}")
    print(f"render_mode: {exp_args.render_mode}")
    print(f"planner_type: {exp_args.planner_type}")
    print(f"map_config: {exp_args.map_config}")
    
    # 检查是否是指定的参数组合
    condition_met = (
        exp_args.run_mode == "hil" and 
        exp_args.robot_mode == "planner" and 
        exp_args.human_mode == "real" and 
        exp_args.human_control == "joystick" and 
        exp_args.render_mode == "gui" and 
        exp_args.planner_type == "flowmatching" and
        "varied_maps_test_holdout.yml" in exp_args.map_config
    )
    
    print(f"\n条件满足: {condition_met}")
    
    if condition_met:
        # 自动设置不使用人类条件的模型路径
        flowmatching_model_path = os.path.join(workspace_root, "table-carrying-ai", "trained_models", "flowmatching", "model_10Hz.ckpt")
        print(f"模型路径: {flowmatching_model_path}")
        print(f"模型文件存在: {os.path.exists(flowmatching_model_path)}")
        
        if os.path.exists(flowmatching_model_path):
            exp_args.artifact_path = flowmatching_model_path
            exp_args.human_act_as_cond = False
            print("✓ 自动设置模型路径")
            print("✓ 设置 human_act_as_cond = False")
        else:
            print("⚠️  警告: 模型文件不存在")

if __name__ == "__main__":
    test_parameter_detection()









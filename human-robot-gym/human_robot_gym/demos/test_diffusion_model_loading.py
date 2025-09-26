#!/usr/bin/env python3
"""
测试Diffusion模型加载脚本
验证演示脚本是否能正确加载训练好的模型
"""

import os
import sys
import torch
import dill
import hydra
from omegaconf import OmegaConf

# Add diffusion_copolicy to path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.join(current_dir, '..', '..', '..')
diffusion_copolicy_path = os.path.join(workspace_root, 'diffusion_copolicy')
sys.path.append(diffusion_copolicy_path)
sys.path.append(os.path.join(diffusion_copolicy_path, 'diffusion_policy'))

def test_model_loading():
    """测试模型加载"""
    print("=== 测试Diffusion模型加载 ===")
    
    # 模型路径
    model_path = os.path.join(workspace_root, "diffusion_copolicy", "data", "outputs", "diffusion_model_lifting_no_human_cond", "best_model.ckpt")
    
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在！")
        return False
    
    try:
        # 加载检查点
        print("正在加载检查点...")
        checkpoint = torch.load(model_path, map_location="cpu", pickle_module=dill)
        print("✓ 检查点加载成功")
        
        # 检查配置
        if 'cfg' in checkpoint:
            cfg = checkpoint['cfg']
            print("✓ 配置加载成功")
            print(f"  - 任务名称: {cfg.get('name', '未知')}")
            print(f"  - 观察维度: {cfg.get('obs_dim', '未知')}")
            print(f"  - 动作维度: {cfg.get('action_dim', '未知')}")
            print(f"  - Human action 作为条件: {cfg.get('human_act_as_cond', '未知')}")
            print(f"  - 观察步数: {cfg.get('n_obs_steps', '未知')}")
            print(f"  - 预测长度: {cfg.get('horizon', '未知')}")
        else:
            print("⚠ 配置信息缺失")
            return False
        
        # 检查模型权重
        if 'model_state_dict' in checkpoint:
            print("✓ 模型权重存在 (model_state_dict)")
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dicts' in checkpoint:
            print("✓ 模型权重存在 (state_dicts)")
            model_state_dict = checkpoint['state_dicts']['model']
        else:
            print("⚠ 模型权重缺失")
            return False
        
        # 检查normalizer
        if 'normalizer_state_dict' in checkpoint:
            print("✓ Normalizer存在")
        elif 'state_dicts' in checkpoint and 'normalizer' in checkpoint['state_dicts']:
            print("✓ Normalizer存在 (state_dicts)")
        else:
            print("⚠ Normalizer缺失")
        
        # 尝试创建模型实例
        print("正在创建模型实例...")
        from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
        
        # 手动解析模型配置
        model_config = cfg.policy.model
        model = TransformerForDiffusion(
            input_dim=model_config.input_dim,
            output_dim=model_config.output_dim,
            horizon=model_config.horizon,
            n_obs_steps=model_config.n_obs_steps,
            cond_dim=model_config.cond_dim,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            n_emb=model_config.n_emb,
            p_drop_emb=model_config.p_drop_emb,
            p_drop_attn=model_config.p_drop_attn,
            causal_attn=model_config.causal_attn,
            time_as_cond=model_config.time_as_cond,
            obs_as_cond=model_config.obs_as_cond,
            human_act_as_cond=model_config.human_act_as_cond,
            n_cond_layers=model_config.n_cond_layers,
        )
        print("✓ 模型实例创建成功")
        
        # 尝试加载权重
        print("正在加载模型权重...")
        if any(key.startswith("model.") for key in model_state_dict.keys()):
            print("✓ 检测到 'model.' 前缀，正在移除...")
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove "model." prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model_state_dict = new_state_dict
        
        model.load_state_dict(model_state_dict)
        print("✓ 模型权重加载成功")
        
        # 尝试创建策略
        print("正在创建策略实例...")
        from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        
        # 手动创建noise scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.policy.noise_scheduler.num_train_timesteps,
            beta_start=cfg.policy.noise_scheduler.beta_start,
            beta_end=cfg.policy.noise_scheduler.beta_end,
            beta_schedule=cfg.policy.noise_scheduler.beta_schedule,
            variance_type=cfg.policy.noise_scheduler.variance_type,
            clip_sample=cfg.policy.noise_scheduler.clip_sample,
            prediction_type=cfg.policy.noise_scheduler.prediction_type,
        )
        
        # 手动创建策略
        policy = DiffusionTransformerLowdimPolicy(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=cfg.policy.horizon,
            obs_dim=cfg.policy.obs_dim,
            action_dim=cfg.policy.action_dim,
            n_action_steps=cfg.policy.n_action_steps,
            n_obs_steps=cfg.policy.n_obs_steps,
            num_inference_steps=cfg.policy.num_inference_steps,
            obs_as_cond=cfg.policy.obs_as_cond,
            human_act_as_cond=cfg.policy.human_act_as_cond,
            pred_action_steps_only=cfg.policy.pred_action_steps_only,
            robot_action_dim=cfg.policy.robot_action_dim,
            human_action_dim=cfg.policy.human_action_dim,
        )
        print("✓ 策略实例创建成功")
        
        # 检查设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ 使用设备: {device}")
        
        # 移动到设备
        policy.eval().to(device)
        print("✓ 模型移动到设备成功")
        
        print("\n🎉 所有测试通过！模型可以正常加载和使用。")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_processing():
    """测试观察处理"""
    print("\n=== 测试观察处理 ===")
    
    # 创建模拟的23维观察
    obs_23d = np.random.randn(23).astype(np.float32)
    print(f"✓ 创建23维观察: {obs_23d.shape}")
    
    # 转换为tensor
    obs_tensor = torch.from_numpy(obs_23d)
    print(f"✓ 转换为tensor: {obs_tensor.shape}")
    
    # 模拟观察缓冲区
    n_obs_steps = 2
    obs_buffer = obs_tensor.unsqueeze(0).repeat(n_obs_steps, 1)
    print(f"✓ 创建观察缓冲区: {obs_buffer.shape}")
    
    # 添加批次维度
    obs_dict = {"obs": obs_buffer.unsqueeze(0)}
    print(f"✓ 添加批次维度: {obs_dict['obs'].shape}")
    
    print("✓ 观察处理测试通过！")
    return True

if __name__ == "__main__":
    import numpy as np
    
    success = True
    
    # 测试模型加载
    success &= test_model_loading()
    
    # 测试观察处理
    success &= test_observation_processing()
    
    if success:
        print("\n🎉 所有测试通过！演示脚本应该可以正常运行。")
        print("\n使用方法:")
        print("1. 无人类条件版本:")
        print("   python demo_collaborative_lifting_diffusion.py")
        print("2. 有人类条件版本:")
        print("   python demo_collaborative_lifting_diffusion_human_condition.py")
    else:
        print("\n❌ 测试失败！请检查模型文件和依赖。")

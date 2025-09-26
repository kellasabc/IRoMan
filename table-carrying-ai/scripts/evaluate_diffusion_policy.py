#!/usr/bin/env python3
import sys
import torch
import dill
import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import shutil

sys.path.append('/home/ubuntu/IRoMan/diffusion_copolicy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.table_lowdim_runner import TableLowdimRunner

def main():
    model_ckpt = "/home/ubuntu/IRoMan/table-carrying-ai/trained_models/diffusion/model_10Hz.ckpt" #model_human_act_as_cond_10Hz.ckpt  model_10Hz.ckpt
    print(f"加载模型: {model_ckpt}")
    payload = torch.load(open(model_ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.eval().to("cuda")
    print("✓ 模型加载完成")

    output_dir = "./eval_outputs"
    os.makedirs(output_dir, exist_ok=True)
    media_dir = os.path.join(output_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    n_eval = 1
    for i in range(n_eval):
        seed = 100 + i
        print(f"\n=== 第{i+1}次评测，种子: {seed} ===")
        runner = TableLowdimRunner(
            output_dir=output_dir,
            n_train=0,
            n_train_vis=0,
            n_test=1,
            n_test_vis=1,
            test_start_seed=seed,
            max_steps=3000,
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            fps=10,
            crf=22,
            past_action= cfg.human_act_as_cond,# cfg.human_act_as_cond,
            map_config="/home/ubuntu/IRoMan/table-carrying-ai/cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        )
        log_data = runner.run(policy)
        # 查找reward和视频
        reward = None
        for key, value in log_data.items():
            if 'sim_max_reward' in key:
                reward = value
        # 查找新生成的视频文件
        files = os.listdir(media_dir)
        mp4_files = sorted([f for f in files if f.endswith('.mp4')], key=lambda x: os.path.getmtime(os.path.join(media_dir, x)))
        if not mp4_files:
            print("未找到视频文件")
            continue
        latest_video = mp4_files[-1]
        src_path = os.path.join(media_dir, latest_video)
        # 判断是否成功
        is_success = reward is not None and reward > -0.1
        if is_success:
            dst_path = os.path.join(media_dir, f"success_{i+1:02d}_{latest_video}")
            print(f"Success! 视频重命名为: {os.path.basename(dst_path)}")
        else:
            dst_path = os.path.join(media_dir, f"fail_{i+1:02d}_{latest_video}")
            print(f"Fail. 视频重命名为: {os.path.basename(dst_path)}")
        shutil.move(src_path, dst_path)

    # 总结
    all_files = sorted(os.listdir(media_dir))
    print("\n=== 所有视频文件 ===")
    for f in all_files:
        print(f)

if __name__ == "__main__":
    main()
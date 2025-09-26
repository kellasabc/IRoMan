#!/usr/bin/env python3
"""
Diffusion Policy Evaluation Script for Table Carrying AI

This script performs evaluation of trained diffusion policy models similar to the
diffusion_copolicy evaluation process, with video recording capabilities.

Usage:
    python evaluate_diffusion_policy.py --run-mode [coplanning | replay_traj | hil] \
        --robot-mode planner --human-mode [planner | data | real] \
        --human-control joystick --planner-type diffusion_policy \
        --map-config cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml \
        --record-video --video-fps 10
"""

import argparse
import pickle
import random
import sys
import time
import os
import pathlib
from os import listdir, mkdir
from os.path import dirname, isdir, isfile, join

import gym
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
import tqdm
import collections

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import experiment configs
from configs.experiment.experiment_config import get_experiment_args
from configs.robot.robot_planner_config import get_planner_args
from cooperative_transport.gym_table.envs.utils import (CONST_DT, WINDOW_H, WINDOW_W)
from libs.hil_methods_diffusion import play_hil_planner

# Add diffusion policy path
diffusion_path = os.path.join(os.path.dirname(project_root), "diffusion_copolicy")
if os.path.exists(diffusion_path):
    sys.path.insert(0, diffusion_path)
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
else:
    print("Warning: diffusion_copolicy path not found, some imports may fail")


    # Define dummy classes for compatibility
    class BaseLowdimPolicy:
        pass


    class BaseWorkspace:
        pass

VERBOSE = False  # Set to True to print debug info


class DiffusionPolicyEvaluator:
    """Diffusion Policy Evaluator with video recording capabilities"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup directories
        self.setup_directories()

        # Load diffusion policy
        self.policy = self.load_diffusion_policy()

        # Setup video recording
        if args.record_video:
            self.setup_video_recording()

    def setup_directories(self):
        """Setup output directories"""
        if not isdir(self.args.results_dir):
            mkdir(self.args.results_dir)

        self.save_dir = join(self.args.results_dir, self.args.exp_name)
        print(f"Saving results to: {self.save_dir}")
        if not isdir(self.save_dir):
            mkdir(self.save_dir)

        # Create video directory if recording
        if self.args.record_video:
            self.video_dir = join(self.save_dir, "videos")
            if not isdir(self.video_dir):
                mkdir(self.video_dir)

    def load_diffusion_policy(self):
        """Load trained diffusion policy model"""
        print("Loading diffusion policy model...")

        # Determine checkpoint path based on human action conditioning
        if self.args.human_act_as_cond:
            ckpt_path = join(self.args.artifact_path, "model_human_act_as_cond_10Hz.ckpt")
            print('Using human actions as conditioning!')
        else:
            ckpt_path = join(project_root, self.args.artifact_path, "model_10Hz.ckpt")
            print('Not using human actions as conditioning!')

            if not isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # Get policy
        policy = workspace.model
        policy.eval().to(self.device)
        policy.num_inference_steps = self.args.num_inference_steps
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

        print(f"Policy loaded successfully. Device: {self.device}")
        return policy

    def setup_video_recording(self):
        """Setup video recording configuration"""
        self.video_fps = self.args.video_fps
        self.video_crf = self.args.video_crf
        print(f"Video recording enabled: {self.video_fps} FPS, CRF: {self.video_crf}")

    def get_test_files(self):
        """Get test trajectory files"""
        files = [
            join(project_root, self.args.data_dir, sd, ssd)
            for sd in listdir(join(project_root, self.args.data_dir))
            if isdir(join(project_root,self.args.data_dir, sd))
            for ssd in listdir(join(project_root,self.args.data_dir, sd))
        ]

        map_files = [
            join(project_root,self.args.map_dir, sd)
            for sd in listdir(join(project_root, self.args.map_dir))
            if isfile(join(project_root, self.args.map_dir, sd))
        ]

        return files, map_files

    def create_environment(self, map_file, episode_name):
        """Create evaluation environment"""
        render_mode = "headless" if self.args.record_video else self.args.render_mode

        env = gym.make(
            "cooperative_transport.gym_table:table-v0",
            render_mode=render_mode,
            control=self.args.human_control,
            map_config=self.args.map_config,
            run_mode="eval",
            load_map=map_file,
            run_name=self.args.exp_name,
            ep=self.args.ep,
            dt=CONST_DT,
        )

        # Create collision checking environment
        collision_checking_env = gym.make(
            "cooperative_transport.gym_table:table-v0",
            render_mode="headless",
            control=self.args.human_control,
            map_config=self.args.map_config,
            run_mode="eval",
            load_map=map_file,
            run_name=self.args.exp_name,
            ep=self.args.ep,
            dt=CONST_DT,
        )

        return env, collision_checking_env

    def run_single_evaluation(self, env, collision_checking_env, playback_trajectory, episode_name):
        """Run single episode evaluation"""
        print(f"Running evaluation for episode: {episode_name}")

        # Run evaluation using hil_methods_diffusion
        trajectory, success, n_iter, duration, avg_planning_time = play_hil_planner(
            env=env,
            exp_run_mode=self.args.run_mode,
            human=self.args.human_mode,
            robot=self.args.robot_mode,
            planner_type=self.args.planner_type,
            artifact_path=self.args.artifact_path,
            mcfg=self.args,
            SEQ_LEN=self.args.SEQ_LEN,
            H=self.args.H,
            skip=self.args.skip,
            num_candidates=self.args.BSIZE,
            playback_trajectory=playback_trajectory,
            collision_checking_env=collision_checking_env,
            display_pred=self.args.display_pred,
            display_gt=self.args.display_gt,
            display_past_states=self.args.display_past_states,
            device=self.device,
            include_interaction_forces_in_rewards=self.args.include_interaction_forces_in_rewards,
        )

        print(f"Episode {episode_name} finished:")
        print(f"  Success: {success}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Steps: {n_iter}")
        print(f"  Avg planning time: {avg_planning_time:.4f} seconds")

        return trajectory, success, n_iter, duration, avg_planning_time

    def save_results(self, trajectory, episode_name, success, n_iter, duration, avg_planning_time):
        """Save evaluation results"""
        if self.args.robot_mode == "data" and self.args.human_mode == "data":
            return

        # Determine robot and human strings for filename
        if self.args.robot_mode == "planner":
            robot_str = f"{self.args.robot_mode}-{self.args.planner_type}"
        else:
            robot_str = self.args.robot_mode

        if self.args.human_mode == "real":
            human_str = f"{self.args.human_mode}-{self.args.human_control}"
        else:
            human_str = self.args.human_mode

        save_filename = f"{self.args.exp_name}-{episode_name}"
        save_path = join(self.save_dir, save_filename)

        # Save trajectory data
        if self.args.planner_type == "rrt":
            pickle.dump(trajectory, open(save_path, "wb"))
        else:
            np.savez(
                save_path,
                states=trajectory["states"],
                plan=trajectory["plan"],
                actions=trajectory["actions"],
                rewards=trajectory["rewards"],
                fluency=trajectory["fluency"],
                done=trajectory["done"],
                success=trajectory["success"],
                n_iter=trajectory["n_iter"],
                duration=trajectory["duration"],
                avg_planning_time=avg_planning_time,
            )

        print(f"Results saved to: {save_path}")

    def run_evaluation(self):
        """Run complete evaluation"""
        print("Starting diffusion policy evaluation...")

        # Get test files
        files, map_files = self.get_test_files()
        print(f"Found {len(files)} test files")

        # Track metrics
        success_count = 0
        total_duration = 0
        total_steps = 0
        planning_times = []

        # Run evaluation for each file
        for f_idx, f in enumerate(files):
            print(f"\n--- Evaluation {f_idx + 1}/{len(files)} ---")

            # Extract episode name
            game_str = f.split("/")
            episode_name = game_str[-1]

            # Find matching map file
            match = [join(project_root, map_file) for map_file in map_files if map_file.split("/")[-1] in episode_name]
            if not match:
                print(f"Warning: No matching map file for {episode_name}")
                continue

            map_file = match[0]

            # Create environment
            env, collision_checking_env = self.create_environment(map_file, episode_name)

            # Load playback trajectory
            playback_trajectory = dict(np.load(f, allow_pickle=True))

            try:
                # Run evaluation
                trajectory, success, n_iter, duration, avg_planning_time = self.run_single_evaluation(
                    env, collision_checking_env, playback_trajectory, episode_name
                )

                # Save results
                self.save_results(trajectory, episode_name, success, n_iter, duration, avg_planning_time)

                # Update metrics
                if success:
                    success_count += 1
                total_duration += duration
                total_steps += n_iter
                planning_times.append(avg_planning_time)

            except Exception as e:
                print(f"Error in episode {episode_name}: {e}")
                continue
            finally:
                # Clean up
                env.close()
                collision_checking_env.close()

        # Print final results
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total episodes: {len(files)}")
        print(f"Successful episodes: {success_count}")
        print(f"Success rate: {success_count / len(files) * 100:.2f}%")
        print(f"Average duration: {total_duration / len(files):.2f} seconds")
        print(f"Average steps: {total_steps / len(files):.1f}")
        if planning_times:
            print(f"Average planning time: {np.mean(planning_times):.4f} seconds")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser("Diffusion Policy Evaluation for Table Carrying AI")

    # Add experiment arguments
    exp_args = parser.add_argument_group("Experiment Settings")
    get_experiment_args(exp_args)

    # Add planner arguments
    planner_args = parser.add_argument_group("Planner Settings")
    get_planner_args(planner_args)

    # Override default values to match the specified command
    exp_args.set_defaults(
        run_mode="replay_traj",#"coplanning",
        robot_mode="planner",
        human_mode="data",
        planner_type="diffusion_policy",
        map_config=join(project_root, "cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml"),
        results_dir=join(project_root, "results"),
        render_mode="human",  # 设置为可视化模式
        #display_pred=True,  # 显示预测轨迹
        #display_gt=True,  # 显示真实轨迹
        #display_past_states=True  # 显示历史状态
    )

    # Set video recording defaults
    diffusion_args = parser.add_argument_group("Diffusion Policy Settings")
    diffusion_args.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Number of inference steps for diffusion sampling"
    )
    diffusion_args.add_argument(
        "--record-video",
        action="store_true",
        default=True,  # Changed from False to True
        help="Record evaluation videos"
    )
    diffusion_args.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Video recording FPS"
    )
    diffusion_args.add_argument(
        "--video-crf",
        type=int,
        default=22,
        help="Video compression quality (lower = better quality)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.run_mode not in ["replay_traj", "hil", "coplanning", "copolicy"]:
        raise ValueError("Run mode not supported")

    if args.run_mode == "hil":
        if args.human_mode == "data":
            raise ValueError("If --human-mode is 'data', --run-mode should be 'replay_traj', not 'hil'")

    if args.robot_mode not in ["planner", "policy", "data"]:
        raise ValueError("Robot mode not supported")

    if args.human_mode not in ["planner", "data", "real"]:
        raise ValueError("Human mode not supported")

    # Create evaluator and run evaluation
    evaluator = DiffusionPolicyEvaluator(args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
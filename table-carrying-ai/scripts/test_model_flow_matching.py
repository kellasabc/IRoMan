import argparse
import pickle
import random
import sys
import time
from os import listdir, mkdir
from os.path import dirname, isdir, isfile, join

import gym
import numpy as np
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.join(current_dir, '..', '..')
sys.path.append(os.path.join(workspace_root, 'flow_copolicy', 'flow_policy'))
sys.path.append(workspace_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.experiment.experiment_config import get_experiment_args
from configs.robot.robot_planner_config import get_planner_args
from cooperative_transport.gym_table.envs.utils import (CONST_DT, WINDOW_H,
                                                        WINDOW_W)
from libs.hil_methods_flow_matching import play_hil_planner

VERBOSE = False


def main(exp_args, exp_name):
    SEED = exp_args.seed
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Set device based on --gpu parameter
    if exp_args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        if exp_args.gpu and not torch.cuda.is_available():
            print("⚠️  Warning: GPU requested but CUDA not available, falling back to CPU")
        else:
            print("✓ Using CPU")

    print("Human actions from: {0}. ".format(exp_args.human_control))
    
    # Check if it's the specified parameter combination, if so automatically set model path
    if (exp_args.run_mode == "hil" and 
        exp_args.robot_mode == "planner" and 
        exp_args.human_mode == "real" and 
        exp_args.human_control == "joystick" and 
        exp_args.render_mode == "gui" and 
        exp_args.planner_type == "flowmatching" and
        "varied_maps_test_holdout.yml" in exp_args.map_config):
        
        # Automatically set model path without human conditions
        flowmatching_model_path = os.path.join(workspace_root, "table-carrying-ai", "trained_models", "flowmatching", "model_10Hz.ckpt")
        if os.path.exists(flowmatching_model_path):
            exp_args.artifact_path = flowmatching_model_path
            # Set to not use human conditions
            exp_args.human_act_as_cond = False
            print(f"✓ Auto-set model path: {flowmatching_model_path}")
            print("✓ Using Flow Matching model without human conditions")
            print("✓ Set human_act_as_cond = False")
        else:
            print(f"⚠️  Warning: Model file does not exist: {flowmatching_model_path}")
            print("Will use default artifact_path setting")

    if not isdir(exp_args.results_dir):
        mkdir(exp_args.results_dir)
    save_dir = join(exp_args.results_dir, exp_name)
    print("Saving results to: {0}".format(save_dir))
    if not isdir(save_dir):
        mkdir(save_dir)

    # Check if dataset is needed (only when human or robot is data)
    need_dataset = (exp_args.human_mode == "data" or exp_args.robot_mode == "data")
    
    success_rate = 0
    
    if need_dataset:
        # Case when dataset is needed
        data_dir_contents = listdir(exp_args.data_dir)
        if any(isdir(join(exp_args.data_dir, item)) for item in data_dir_contents):
            FILES = [
                join(exp_args.data_dir, sd, ssd)
                for sd in listdir(exp_args.data_dir)
                if isdir(join(exp_args.data_dir, sd))
                for ssd in listdir(join(exp_args.data_dir, sd))
            ]
        else:
            FILES = [
                join(exp_args.data_dir, f)
                for f in listdir(exp_args.data_dir)
                if isfile(join(exp_args.data_dir, f)) and f.endswith('.pkl')
            ]

        MAP_FILES = [
            join(exp_args.map_dir, sd)
            for sd in listdir(exp_args.map_dir)
            if isfile(join(exp_args.map_dir, sd))
        ]

        for f_idx in range(len(FILES)):
            f = FILES[f_idx]
            game_str = f.split("/")
            ep = game_str[-1]
            robot_control = exp_args.robot_mode
            match = [map for map in MAP_FILES if os.path.splitext(map.split("/")[-1])[0] in ep]

            env = gym.make(
                "cooperative_transport.gym_table:table-v0",
                render_mode=exp_args.render_mode,
                control=exp_args.human_control,
                map_config=exp_args.map_config,
                run_mode="eval",
                load_map=match[0],
                run_name=exp_name,
                ep=exp_args.ep,
                dt=CONST_DT,
            )

            collision_checking_env = gym.make(
                "cooperative_transport.gym_table:table-v0",
                render_mode=exp_args.render_mode,
                control=exp_args.human_control,
                map_config=exp_args.map_config,
                run_mode="eval",
                load_map=match[0],
                run_name=exp_name,
                ep=exp_args.ep,
                dt=CONST_DT,
            )

            trajectory, success, n_iter, duration, avg_planning_time = play_hil_planner(
                env,
                exp_run_mode=exp_args.run_mode,
                human=exp_args.human_mode,
                robot=robot_control,
                planner_type="flow_matching",
                artifact_path=exp_args.artifact_path,
                mcfg=exp_args,
                SEQ_LEN=exp_args.SEQ_LEN,
                H=exp_args.H,
                skip=exp_args.skip,
                num_candidates=exp_args.BSIZE,
                playback_trajectory=np.load(f, allow_pickle=True),
                collision_checking_env=collision_checking_env,
                display_pred=exp_args.display_pred,
                display_gt=exp_args.display_gt,
                display_past_states=exp_args.display_past_states,
                include_interaction_forces_in_rewards=exp_args.include_interaction_forces_in_rewards,
                n_steps=3000,
                device=device,  # Pass device parameter
            )

            print(
                "Run finished. Task succeeded: {0}. Duration: {1} sec. Num steps taken in env: {2}. Episode {3}.".format(
                    success, duration, n_iter, ep
                )
            )
            print("Average planning time per planning loop: {0:.4f} sec".format(avg_planning_time))

            if not (exp_args.human_mode == "data" and exp_args.robot_mode == "data"):
                robot_str = exp_args.robot_mode + "-flow_matching"
                if exp_args.human_mode == "real":
                    human_str = exp_args.human_mode + "-" + exp_args.human_control
                else:
                    human_str = exp_args.human_mode

                save_f = exp_name + "-" + ep

                np.savez(
                    join(save_dir, save_f),
                    states=trajectory["states"],
                    plan=trajectory["plan"],
                    actions=trajectory["actions"],
                    rewards=trajectory["rewards"],
                    fluency=trajectory["fluency"],
                    done=trajectory["done"],
                    success=trajectory["success"],
                    n_iter=trajectory["n_iter"],
                    duration=trajectory["duration"],
                )

                success_rate += trajectory["success"]
        success_rate /= len(FILES)
        print("Success rate: {0}, with {1} total trials.".format(success_rate, len(FILES)))
    else:
        # Case when dataset is not needed (HIL real human interaction)
        # Loop through multiple episodes, similar to test_model.py
        # Default: infinite loop, unless --finite-loop is specified or --num-episodes is positive
        
        def run_episode(episode_num):
            """Run a single episode"""
            print(f"\n=== Episode {episode_num} ===")
            
            env = gym.make(
                "cooperative_transport.gym_table:table-v0",
                render_mode=exp_args.render_mode,
                control=exp_args.human_control,
                map_config=exp_args.map_config,
                run_mode="eval",
                load_map=None,
                run_name=exp_name,
                ep=episode_num - 1,  # Use episode number (0-indexed)
                dt=CONST_DT,
            )

            collision_checking_env = gym.make(
                "cooperative_transport.gym_table:table-v0",
                render_mode=exp_args.render_mode,
                control=exp_args.human_control,
                map_config=exp_args.map_config,
                run_mode="eval",
                load_map=None,
                run_name=exp_name,
                ep=episode_num - 1,  # Use episode number (0-indexed)
                dt=CONST_DT,
            )

            trajectory, success, n_iter, duration, avg_planning_time = play_hil_planner(
                env,
                exp_run_mode=exp_args.run_mode,
                human=exp_args.human_mode,
                robot=exp_args.robot_mode,
                planner_type="flow_matching",
                artifact_path=exp_args.artifact_path,
                mcfg=exp_args,
                SEQ_LEN=exp_args.SEQ_LEN,
                H=exp_args.H,
                skip=exp_args.skip,
                num_candidates=exp_args.BSIZE,
                playback_trajectory=None,
                collision_checking_env=collision_checking_env,
                display_pred=exp_args.display_pred,
                display_gt=exp_args.display_gt,
                display_past_states=exp_args.display_past_states,
                include_interaction_forces_in_rewards=exp_args.include_interaction_forces_in_rewards,
                n_steps=3000,
                device=device,  # Pass device parameter
            )

            print(
                "Episode {0} finished. Task succeeded: {1}. Duration: {2} sec. Num steps taken in env: {3}.".format(
                    episode_num, success, duration, n_iter
                )
            )
            print("Average planning time per planning loop: {0:.4f} sec".format(avg_planning_time))

            # Save results for each episode
            if not (exp_args.human_mode == "data" and exp_args.robot_mode == "data"):
                robot_str = exp_args.robot_mode + "-flow_matching"
                if exp_args.human_mode == "real":
                    human_str = exp_args.human_mode + "-" + exp_args.human_control
                else:
                    human_str = exp_args.human_mode

                save_f = exp_name + f"-episode_{episode_num}"

                np.savez(
                    join(save_dir, save_f),
                    states=trajectory["states"],
                    plan=trajectory["plan"],
                    actions=trajectory["actions"],
                    rewards=trajectory["rewards"],
                    fluency=trajectory["fluency"],
                    done=trajectory["done"],
                    success=trajectory["success"],
                    n_iter=trajectory["n_iter"],
                    duration=trajectory["duration"],
                )

            # Clean up environment
            env.close()
            collision_checking_env.close()
            
            return float(trajectory["success"]) if not isinstance(trajectory["success"], (list, np.ndarray)) else float(np.array(trajectory["success"]).mean())
        
        # Main loop logic
        if exp_args.finite_loop or exp_args.num_episodes > 0:
            # Finite loop
            num_episodes = exp_args.num_episodes if exp_args.num_episodes > 0 else 5
            print(f"Running {num_episodes} episodes...")
            
            for episode_num in range(1, num_episodes + 1):
                episode_success = run_episode(episode_num)
                success_rate += episode_success
            
            success_rate /= num_episodes
            print(f"\n=== Final Results ===")
            print("Success rate: {0:.2f}, with {1} total trials.".format(success_rate, num_episodes))
        else:
            # Default: infinite loop
            print("Running infinite episodes (press Ctrl+C to stop)...")
            episode_num = 1
            total_episodes = 0
            
            try:
                while True:
                    episode_success = run_episode(episode_num)
                    success_rate += episode_success
                    total_episodes += 1
                    episode_num += 1
                    
                    # Print running statistics every episode
                    avg_success = success_rate / total_episodes
                    print(f"\n=== Running Statistics (Last {total_episodes} episodes) ===")
                    print("Average success rate: {0:.2f}".format(avg_success))
            except KeyboardInterrupt:
                print(f"\n=== Stopped by user ===")
                if total_episodes > 0:
                    avg_success = success_rate / total_episodes
                    print("Final success rate: {0:.2f}, with {1} total trials.".format(avg_success, total_episodes))
                else:
                    print("No episodes completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Table Carrying Experiments (Flow Matching).")

    exp_args = parser.add_argument_group("Experiment Settings")
    get_experiment_args(exp_args)
    
    # Add GPU parameter
    exp_args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for computation (if available)"
    )
    
    # Add episode count parameter
    exp_args.add_argument(
        "--num-episodes",
        type=int,
        default=-1,
        help="Number of episodes to run (default: -1 for infinite loop, use positive number for finite episodes)"
    )
    
    # Add finite loop parameter (opposite of infinite)
    exp_args.add_argument(
        "--finite-loop",
        action="store_true",
        help="Run finite number of episodes instead of infinite loop"
    )
    
    # Always add planner_args because we need to support --planner-type and --human-act-as-cond
    get_planner_args(exp_args)
    
    # Parse arguments
    exp_args = parser.parse_args()
    
    # Validate run_mode
    assert exp_args.run_mode in [
        "replay_traj",
        "hil",
        "coplanning",
        "copolicy",
    ], "Run mode not supported"
    
    if exp_args.run_mode == "hil":
        assert (
            exp_args.human_mode != "data"
        ), "If --human-mode is 'data', --run-mode should be 'replay_traj', not 'hil'"

    # Validate robot_mode and human_mode
    if exp_args.robot_mode == "policy" or exp_args.human_mode == "policy":
        pass
    elif exp_args.robot_mode == "data" or exp_args.human_mode == "data":
        assert exp_args.run_mode == "replay_traj", "If --robot-mode or --human-mode is 'data', --run-mode should be 'replay_traj', not 'hil'"
    elif exp_args.robot_mode != "planner" and exp_args.human_mode != "planner" and exp_args.run_mode != "replay_traj":
        raise ValueError("Robot mode not supported")

    print("Begin experiment in {} mode!".format(exp_args.run_mode))

    if exp_args.run_mode == "hil":
        assert (
            not exp_args.human_mode == "planner"
        ), "Set --run_mode to coplanning if both robot and human is planner"
        assert not (
            exp_args.human_mode == "data"
        ), "HIL mode requires human behaviors not from data (i.e. not open-loop). Set --run_mode to replay_traj instead."

    if exp_args.robot_mode == "planner":
        robot_str = exp_args.robot_mode + "-flow_matching"
    else:
        robot_str = exp_args.robot_mode
    human_str = exp_args.human_mode + "-" + exp_args.human_control

    exp_name = (
        "eval_"
        + exp_args.run_mode
        + "_seed-"
        + str(exp_args.seed)
        + "_R-"
        + robot_str
        + "_H-"
        + human_str
    )

    print("Experiment name: ", exp_name)

    main(exp_args, exp_name)



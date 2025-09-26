import sys
import time
from os.path import dirname, join

import numpy as np
import pygame
import torch

from cooperative_transport.gym_table.envs.custom_rewards import \
    custom_reward_function
from cooperative_transport.gym_table.envs.utils import (CONST_DT, FPS,
                                                        MAX_FRAMESKIP,
                                                        WINDOW_H, WINDOW_W,
                                                        debug_print,
                                                        get_keys_to_action,
                                                        init_joystick,
                                                        set_action_keyboard)
from libs.planner.planner_utils import (is_safe, pid_single_step, tf2model,
                                        tf2sim, update_queue)

FPS = 30
new_fps = 10
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 3
SKIP_FRAME = FPS // new_fps


def compute_reward(
    states,
    goal,
    obs,
    env=None,
    vectorized=False,
    interaction_forces=None,
    u_r=None,
    u_h=None,
    collision=None,
    collision_checking_env=None,
    success=None,
) -> float:
    if env.include_interaction_forces_in_rewards:
        reward = custom_reward_function(
            states,
            goal,
            obs,
            interaction_forces=interaction_forces,
            vectorized=True,
            collision_checking_env=collision_checking_env,
            env=env,
            u_h=u_h,
        )
    else:
        reward = custom_reward_function(
            states, goal, obs, vectorized=True, env=env, collision_checking_env=collision_checking_env, u_h=u_h
        )
    return reward


def play_hil_planner(
    env,
    exp_run_mode="hil",
    human="data",
    robot="planner",
    planner_type="flow_matching",
    artifact_path=None,
    mcfg=None,
    SEQ_LEN=120,
    H=30,
    skip=5,
    num_candidates=64,
    playback_trajectory=None,
    n_steps=1000,
    fps=FPS,
    collision_checking_env=None,
    display_pred=False,
    display_gt=False,
    display_past_states=False,
    device="cpu",
    include_interaction_forces_in_rewards=False,
):
    """
    Flow Matching version of Human-in-the-Loop (HIL) evaluation entry.
    References the overall flow of libs/hil_methods_diffusion.py, but model loading and inference uses flow_copolicy.
    """
    trajectory = {}
    trajectory["states"] = []
    trajectory["plan"] = []
    trajectory["actions"] = []
    trajectory["rewards"] = []
    trajectory["fluency"] = []

    assert human in ["data", "real", "policy", "planner"], (
        "human arg must be one of 'data', 'policy', or 'real'"
    )
    if human == "real":
        if env.control_type == "joystick":
            joysticks = init_joystick()
            p2_id = 0
        elif env.control_type == "keyboard":
            keys_to_action = get_keys_to_action()
            relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
            pressed_keys = []
        else:
            raise ValueError("control_type must be 'joystick' or 'keyboard'")
    elif human == "policy":
        raise NotImplementedError("BC policy not implemented yet")
    elif human == "planner":
        assert (
            robot == "planner" and exp_run_mode == "coplanning"
        ), "Must be in co-planning mode if human is planner."
    else:
        assert playback_trajectory is not None, "Must provide playback trajectory"
        if len(playback_trajectory["actions"].shape) == 3:
            playback_trajectory["actions"] = playback_trajectory["actions"].squeeze()
        assert human == "data"
    if human == "data" or robot == "data":
        n_steps = len(playback_trajectory["actions"]) - 1
    coplanning = True if (human == "planner" and robot == "planner") else False

    assert robot in ["planner", "data", "real"], (
        "robot arg must be one of 'planner' or 'data' or 'real'"
    )
    if robot == "real":
        p1_id = 1

    if robot == "planner":
        # flow matching planner
        if planner_type == "flow_matching":
            import copy
            import os
            import hydra
            import dill
            from omegaconf import OmegaConf

            current_dir = dirname(__file__)
            workspace_root = join(current_dir, "..", "..")
            sys.path.append(join(workspace_root, "flow_copolicy"))
            sys.path.append(join(workspace_root, "flow_copolicy", "flow_policy"))

            from flow_policy.gym_util.multistep_wrapper import MultiStepWrapper
            from flow_policy.gym_util.video_recording_wrapper import (
                VideoRecordingWrapper,
                VideoRecorder,
            )
            from gym.wrappers import FlattenObservation

            # Select corresponding model file based on human_act_as_cond
            if mcfg.human_act_as_cond:
                ckpt_path = join(workspace_root, "table-carrying-ai", "trained_models", "flowmatching", "model_human_act_as_cond_10Hz.ckpt")
                print('✓ Using Flow Matching model with human actions as conditions')
            else:
                ckpt_path = join(workspace_root, "table-carrying-ai", "trained_models", "flowmatching", "model_10Hz.ckpt")
                print('✓ Using Flow Matching model without human actions as conditions')
            
            # If the specified model file doesn't exist, try to use artifact_path
            if not os.path.exists(ckpt_path) and artifact_path is not None and len(str(artifact_path)) > 0 and os.path.exists(artifact_path):
                if os.path.isfile(artifact_path):
                    ckpt_path = artifact_path
                    print(f"✓ Using specified artifact_path: {os.path.basename(ckpt_path)}")
                elif os.path.isdir(artifact_path):
                    # If it's a directory, try to find checkpoint file in it
                    cands = [f for f in os.listdir(artifact_path) if (f.endswith(".pt") or f.endswith(".ckpt"))]
                    if len(cands) > 0:
                        cands = sorted(cands)
                        ckpt_path = join(artifact_path, cands[-1])
                        print(f"✓ Found checkpoint in specified directory: {os.path.basename(ckpt_path)}")
                    else:
                        raise FileNotFoundError(f"Checkpoint file not found in specified directory: {artifact_path}")
                else:
                    raise FileNotFoundError(f"Specified artifact_path does not exist: {artifact_path}")
            
            # If still can't find model file, try to load from training output directory
            if not os.path.exists(ckpt_path):
                if mcfg.human_act_as_cond:
                    default_dir = join(workspace_root, "flow_copolicy", "outputs", "flow_matching_human_cond", "checkpoints")
                    print("✓ Trying to load model from human-conditioned training output directory")
                else:
                    default_dir = join(workspace_root, "flow_copolicy", "outputs", "flow_matching_no_human_cond", "checkpoints")
                    print("✓ Trying to load model from standard training output directory")
                
                # Find the latest checkpoint file
                if os.path.isdir(default_dir):
                    cands = [f for f in os.listdir(default_dir) if (f.endswith(".pt") or f.endswith(".ckpt"))]
                    if len(cands) > 0:
                        cands = sorted(cands)
                        ckpt_path = join(default_dir, cands[-1])
                        print(f"✓ Found checkpoint: {os.path.basename(ckpt_path)}")
                    else:
                        raise FileNotFoundError(f"Checkpoint file not found in training output directory: {default_dir}")
                else:
                    raise FileNotFoundError(f"Training output directory does not exist: {default_dir}")
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Flow Matching model checkpoint not found: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
            cfg = checkpoint["cfg"]

            # Check if it's a diffusion_policy checkpoint, if so replace module paths
            if "diffusion_policy" in str(cfg.policy.model._target_):
                print("✓ Detected diffusion_policy checkpoint, replacing with flow_policy module paths")
                # Replace module paths
                cfg.policy.model._target_ = cfg.policy.model._target_.replace("diffusion_policy", "flow_policy")
                cfg.policy.model._target_ = cfg.policy.model._target_.replace("transformer_for_diffusion.TransformerForDiffusion", "flow_matching_transformer.FlowMatchingTransformer")
                cfg.policy._target_ = cfg.policy._target_.replace("diffusion_policy", "flow_policy")
                cfg.policy._target_ = cfg.policy._target_.replace("diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy", "flow_matching_transformer_lowdim_policy.FlowMatchingTransformerLowdimPolicy")
                print(f"✓ Updated model._target_: {cfg.policy.model._target_}")

            # Set configuration based on mcfg.human_act_as_cond
            if hasattr(mcfg, 'human_act_as_cond'):
                human_act_as_cond = mcfg.human_act_as_cond
                OmegaConf.update(cfg, "human_act_as_cond", human_act_as_cond, merge=False)
                OmegaConf.update(cfg, "task.dataset.human_act_as_cond", human_act_as_cond, merge=False)
                OmegaConf.update(cfg, "policy.human_act_as_cond", human_act_as_cond, merge=False)
                print(f"✓ Set human_act_as_cond = {human_act_as_cond}")
            else:
                # Default to not using human conditions
                human_act_as_cond = False
                OmegaConf.update(cfg, "human_act_as_cond", human_act_as_cond, merge=False)
                OmegaConf.update(cfg, "task.dataset.human_act_as_cond", human_act_as_cond, merge=False)
                OmegaConf.update(cfg, "policy.human_act_as_cond", human_act_as_cond, merge=False)
                print("✓ Default set human_act_as_cond = False")

            from flow_policy.model.diffusion.flow_matching_transformer import FlowMatchingTransformer
            from flow_policy.policy.flow_matching_transformer_lowdim_policy import FlowMatchingTransformerLowdimPolicy

            # Create model instance
            model: FlowMatchingTransformer = hydra.utils.instantiate(cfg.policy.model)
            
            # Load model weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                normalizer_state_dict = checkpoint.get("normalizer_state_dict")
            elif "state_dicts" in checkpoint:
                model_state_dict = checkpoint["state_dicts"]["model"]
                # Check if "model." prefix needs to be removed
                if any(key.startswith("model.") for key in model_state_dict.keys()):
                    print("✓ Detected 'model.' prefix, removing...")
                    new_state_dict = {}
                    for key, value in model_state_dict.items():
                        if key.startswith("model."):
                            new_key = key[6:]  # Remove "model." prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    model_state_dict = new_state_dict
                model.load_state_dict(model_state_dict)
                normalizer_state_dict = checkpoint["state_dicts"].get("normalizer")
            else:
                raise ValueError("Unknown checkpoint format")

            # Create policy
            policy: FlowMatchingTransformerLowdimPolicy = hydra.utils.instantiate(
                cfg.policy,
                model=model,
            )
            
            # Load normalizer (if exists)
            if normalizer_state_dict is not None:
                policy.normalizer.load_state_dict(normalizer_state_dict)
                print("✓ Successfully loaded normalizer")
            else:
                print("⚠ Normalizer not found, using default initialization")

            # Configure environment parameters
            a_horizon = 8
            a_horizon_ct = 0
            zoh_ct = 1

            # Move model to specified device
            policy.eval().to(device)
            print(f"✓ Model moved to device: {device}")
            policy.num_inference_steps = cfg.policy.get("num_inference_steps", 50)
            policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

            # Wrap environment
            steps_per_render = max(10 // FPS, 1)
            def env_fn():
                return MultiStepWrapper(
                    VideoRecordingWrapper(
                        FlattenObservation(env),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec="h264",
                            input_pix_fmt="rgb24",
                            crf=22,
                            thread_type="FRAME",
                            thread_count=1,
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render,
                    ),
                    n_obs_steps=cfg.n_obs_steps,
                    n_action_steps=cfg.n_action_steps,
                    max_episode_steps=2000,
                )
            env = env_fn()

        else:
            raise ValueError("planner_type must be 'flow_matching'")

    # reset environment
    obs = env.reset()
    obs_model = np.tile(np.copy(obs), (SKIP_FRAME, 1))
    past_action = np.zeros(obs_model.shape)[..., :4]

    print("Warming up policy inference")
    with torch.no_grad():
        policy.reset()
        obs_dict_np = {"obs": obs.astype(np.float32)}
        if mcfg.human_act_as_cond:
            obs_dict_np["past_action"] = past_action.astype(np.float32)
        # Simple torch conversion
        def to_tensor_batch(x):
            return torch.from_numpy(x).unsqueeze(0).to(next(policy.model.parameters()).device)
        obs_dict = {k: to_tensor_batch(v) for k, v in obs_dict_np.items()}
        result = policy.predict_action(obs_dict)
        action = result["action"][0]
        assert action.shape[-1] == 4
        del result
    print("Ready!")
    policy.reset()

    if not isinstance(obs, torch.Tensor):
        obs_t = torch.from_numpy(obs).float()
    info = None
    done = False
    n_iter = 0
    running = True
    clock = pygame.time.Clock()
    success = False
    delta_plan_sum = 0
    plan_cter = 0

    if display_past_states:
        past_states = []
        past_states.append(obs.tolist())
    if display_gt:
        waypoints_true = playback_trajectory["states"].tolist() if playback_trajectory is not None else []

    action_plan = None
    start = time.time()
    next_game_tick = time.time()

    while running:
        loops = 0
        if done:
            pygame.quit()
            print("Episode finished after {} timesteps".format(n_iter + 1))
            break
        else:
            start_plan = time.time()

            if display_gt and playback_trajectory is not None:
                env.draw_gt(waypoints_true)

            if human == "real":
                if env.control_type == "joystick":
                    u_h = np.array([
                        joysticks[p2_id].get_axis(0),
                        joysticks[p2_id].get_axis(1),
                    ])
                    u_h = torch.from_numpy(np.clip(u_h, -1.0, 1.0)).unsqueeze(0)
                else:
                    u_h = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                    u_h = set_action_keyboard(u_h)
                    u_h = torch.from_numpy(u_h[1, :]).unsqueeze(0)
            elif human == "planner":
                pass
            else:
                assert human == "data"
                n_iter = min(n_iter, playback_trajectory["actions"].shape[0] - 1)
                u_h = playback_trajectory["actions"][n_iter, 2:]
                u_h = torch.from_numpy(u_h).unsqueeze(0)

            if (a_horizon_ct < a_horizon) and n_iter != 0 and planner_type == "flow_matching":
                # Use previously predicted actions
                u_r = torch.from_numpy(action_plan[a_horizon_ct, :2]).unsqueeze(0)
                if coplanning:
                    u_h = torch.from_numpy(action_plan[a_horizon_ct, 2:]).unsqueeze(0)
                u_all = torch.cat((u_r, u_h), dim=-1)
                # update past_action
                past_action[:-1, :] = past_action[1:, :]
                past_action[-1, :2] = action_plan[a_horizon_ct, :2].squeeze()
                past_action[-1, 2:] = u_h.flatten() if not coplanning else action_plan[a_horizon_ct, 2:]
                a_horizon_ct += 1
                zoh_ct += 1
            else:
                # (re)plan using Flow Matching policy
                a_horizon_ct = 0
                zoh_ct = 1
                obs_dict_np = {"obs": obs_model[::SKIP_FRAME, ...].astype(np.float32)}
                if mcfg.human_act_as_cond:
                    obs_dict_np["past_action"] = past_action[::SKIP_FRAME, ...].astype(np.float32)
                # Use the passed device parameter instead of getting from model
                obs_dict = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs_dict_np.items()}

                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                    plan_cter += 1
                    action_plan = result["action"][0].detach().to("cpu").numpy()

                    u_r = torch.from_numpy(action_plan[a_horizon_ct, :2]).unsqueeze(0)
                    if coplanning:
                        u_h = torch.from_numpy(action_plan[a_horizon_ct, 2:]).unsqueeze(0)
                    u_all = torch.cat((u_r, u_h), dim=-1)

                past_action[:-1, :] = past_action[1:, :]
                past_action[-1, :2] = action_plan[a_horizon_ct, :2].squeeze()
                past_action[-1, 2:] = u_h.flatten() if not coplanning else action_plan[a_horizon_ct, 2:]

                a_horizon_ct += 1
                zoh_ct += 1

                delta_plan = time.time() - start_plan
                delta_plan_sum += delta_plan

            # step env
            obs, reward, done, info = env.step(u_all.unsqueeze(0).detach().to("cpu").numpy())

            # update obs_model
            obs_model[:-1, ...] = obs_model[1:, ...]
            obs_model[-1, ...] = obs[-1, ...]

            n_iter += 1
            if display_past_states:
                past_states.append(obs.tolist())
                env.draw_past_states(past_states)

            obs_t = torch.from_numpy(obs).float()

            trajectory["states"].append(obs_t[-1, ...])
            if robot == "planner":
                if planner_type == "flow_matching":
                    trajectory["plan"].append(action_plan[a_horizon_ct].tolist())
            trajectory["actions"].append(u_all)
            trajectory["rewards"].append(torch.tensor(reward))
            trajectory["fluency"].append(env.fluency)

            if done:
                if info["success"][-1]:
                    success = True
                else:
                    success = False
                env.render(mode="human")
                running = False
                break

            next_game_tick += CONST_DT
            loops += 1
            if not done:
                env.redraw()
                clock.tick(FPS)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if env.control_type == "keyboard" and human == "real":
                            debug_print("REGISTERED KEY PRESS")
                            pressed_keys.append(event.key)
                        elif event.key == 27:
                            running = False
                    elif event.type == pygame.KEYUP:
                        if env.control_type == "keyboard" and human == "real":
                            pressed_keys.remove(event.key)
                    elif event.type == pygame.QUIT:
                        running = False

    stop = time.time()
    duration = stop - start
    print("Duration of run: ", duration)
    pygame.quit()

    if not (human == "data" and robot == "data"):
        trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
        if robot == "planner":
            if planner_type == "flow_matching":
                # Convert list to numpy array
                trajectory["plan"] = np.array(trajectory["plan"])
        trajectory["actions"] = torch.stack(trajectory["actions"], dim=0).numpy()
        trajectory["rewards"] = torch.stack(trajectory["rewards"], dim=0).numpy()
        assert info is not None, "Error: info is None"
        trajectory["fluency"] = info["fluency"]
        # Use the final success state instead of the array
        if isinstance(info["success"], (list, np.ndarray)):
            trajectory["success"] = info["success"][-1] if len(info["success"]) > 0 else False
        else:
            trajectory["success"] = info["success"]
        trajectory["done"] = torch.FloatTensor(np.array([[float(done)],]))
        trajectory["n_iter"] = n_iter
        trajectory["duration"] = duration

    return trajectory, success, n_iter, duration, (delta_plan_sum / max(plan_cter, 1))



import collections
import math
import pathlib
import dill
import numpy as np
import torch
import tqdm
import wandb.sdk.data_types.video as wv
import wandb
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder, VideoRecordingWrapper)
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

# Add human-robot-gym to path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.join(current_dir, '..', '..', '..', '..')
human_robot_gym_path = os.path.join(workspace_root, 'human-robot-gym')
sys.path.append(human_robot_gym_path)

try:
    import robosuite as suite
    from human_robot_gym.utils.env_util import ExpertObsWrapper
    from human_robot_gym.wrappers.visualization_wrapper import VisualizationWrapper
    from human_robot_gym.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper
    from human_robot_gym.wrappers.ik_position_delta_wrapper import IKPositionDeltaWrapper
    from human_robot_gym.utils.mjcf_utils import file_path_completion, merge_configs
    from robosuite.controllers import load_controller_config
    from human_robot_gym.utils.cart_keyboard_controller import KeyboardControllerAgentCart
    import human_robot_gym.robots  # noqa: F401
    HUMAN_ROBOT_GYM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import human-robot-gym modules: {e}")
    print("LiftingLowdimRunner will not work without human-robot-gym")
    HUMAN_ROBOT_GYM_AVAILABLE = False


class LiftingLowdimRunner(BaseLowdimRunner):
    """
    Environment runner for collaborative lifting task using human-robot-gym.
    """
    def __init__(
        self,
        output_dir,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=5,
        crf=22,
        past_action=False,
        abs_action=False,
        obs_eef_target=True,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 10 # Assuming control_freq of 10Hz for the environment
        steps_per_render = max(task_fps // fps, 1)

        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.past_action = past_action
        self.abs_action = abs_action
        self.obs_eef_target = obs_eef_target
        self.tqdm_interval_sec = tqdm_interval_sec
        
        # Mark environment as unavailable
        self.env_available = HUMAN_ROBOT_GYM_AVAILABLE

    def _create_env(self, seed, enable_render=False):
        """Create a single collaborative lifting environment"""
        if not HUMAN_ROBOT_GYM_AVAILABLE:
            print("Human-robot-gym not available, skipping environment creation")
            return None
            
        try:
            # Setup controller configuration
            pybullet_urdf_file = file_path_completion(
                "models/assets/robots/schunk/robot_pybullet.urdf"
            )
            controller_config = dict()
            controller_conig_path = file_path_completion(
                "controllers/failsafe_controller/config/failsafe.json"
            )
            robot_conig_path = file_path_completion("models/robots/config/schunk.json")
            controller_config = load_controller_config(custom_fpath=controller_conig_path)
            robot_config = load_controller_config(custom_fpath=robot_conig_path)
            controller_config = merge_configs(controller_config, robot_config)
            controller_configs = [controller_config]

            rsenv = suite.make(
                "CollaborativeLiftingCart",
                robots="Schunk",  # use Schunk robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=enable_render,  # make sure we can render to the screen
                render_camera=None,
                render_collision_mesh=False,
                control_freq=task_fps,  # control should happen fast enough so that simulation looks smooth
                hard_reset=False,
                horizon=self.max_steps,
                done_at_success=False,
                controller_configs=controller_configs,
                shield_type="SSM",  # Shield mode, can be "SSM" or "PFL"
                visualize_failsafe_controller=False,
                visualize_pinocchio=False,
                base_human_pos_offset=[0.0, 0.0, 0.0],
                human_rand=[0, 0.0, 0.0], # Fixed human animation
                verbose=False,
                human_animation_freq=20,
            )

            env = ExpertObsWrapper(
                env=rsenv,
                agent_keys=[
                    "vec_eef_to_human_head",
                    "vec_eef_to_human_lh",
                    "vec_eef_to_human_rh",
                ],
                expert_keys=[
                    "vec_eef_to_human_lh",
                    "vec_eef_to_human_rh",
                    "board_quat",
                    "board_gripped",
                ]
            )
            env = CollisionPreventionWrapper(
                env=env, collision_check_fn=env.check_collision_action, replace_type=0,
            )
            action_limits = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
            env = IKPositionDeltaWrapper(env=env, urdf_file=pybullet_urdf_file, action_limits=action_limits)

            # Wrap with VideoRecordingWrapper if rendering is enabled
            if enable_render:
                env = VideoRecordingWrapper(
                    env=env,
                    video_recoder=VideoRecorder.create_h264(
                        fps=self.fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=self.crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None, # Set dynamically later
                    steps_per_render=steps_per_render,
                )
            
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                max_episode_steps=self.max_steps,
            )
            return env
        except Exception as e:
            print(f"Error creating lifting environment: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_obs_23d(self, raw_obs):
        """Extract 23-dimensional observation from raw environment observation"""
        obs_23d = []
        
        # robot0_eef_pos (3D)
        obs_23d.extend(raw_obs['robot0_eef_pos'])
        
        # robot0_gripper_qpos (2D) 
        obs_23d.extend(raw_obs['robot0_gripper_qpos'])
        
        # robot0_gripper_qvel (2D)
        obs_23d.extend(raw_obs['robot0_gripper_qvel'])
        
        # vec_eef_to_human_head (3D)
        obs_23d.extend(raw_obs['vec_eef_to_human_head'])
        
        # vec_eef_to_human_lh (3D)
        obs_23d.extend(raw_obs['vec_eef_to_human_lh'])
        
        # vec_eef_to_human_rh (3D)
        obs_23d.extend(raw_obs['vec_eef_to_human_rh'])
        
        # board_quat (4D)
        obs_23d.extend(raw_obs['board_quat'])
        
        # board_balance (1D)
        if 'board_balance' in raw_obs:
            obs_23d.append(raw_obs['board_balance'])
        else:
            obs_23d.append(0.0)  # Default value
        
        # board_gripped (1D)
        if 'board_gripped' in raw_obs:
            obs_23d.append(float(raw_obs['board_gripped']))
        else:
            obs_23d.append(0.0)  # Default value
        
        # dist_eef_to_human_head (1D)
        if 'dist_eef_to_human_head' in raw_obs:
            obs_23d.append(raw_obs['dist_eef_to_human_head'])
        else:
            # Calculate distance from vec_eef_to_human_head
            vec_to_head = raw_obs['vec_eef_to_human_head']
            dist_to_head = np.linalg.norm(vec_to_head)
            obs_23d.append(dist_to_head)
        
        # Total: 3+2+2+3+3+3+4+1+1+1 = 23 dimensions
        return np.array(obs_23d, dtype=np.float32)

    def run(self, policy: BaseLowdimPolicy):
        """Run evaluation with the given policy"""
        # If environment is unavailable, return empty results
        if not self.env_available:
            print("Environment unavailable, skipping environment evaluation")
            return {
                'train_mean_score': 0.0,
                'test_mean_score': 0.0,
                'train_mean_success': 0.0,
                'test_mean_success': 0.0,
            }
        
        device = policy.device
        dtype = policy.dtype

        # Create environments for training and testing
        train_envs = []
        test_envs = []
        
        # Training environments
        for i in range(self.n_train):
            seed = self.train_start_seed + i
            enable_render = i < self.n_train_vis
            env = self._create_env(seed, enable_render)
            if env is not None:
                train_envs.append(env)
        
        # Testing environments
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            enable_render = i < self.n_test_vis
            env = self._create_env(seed, enable_render)
            if env is not None:
                test_envs.append(env)

        # Run evaluation
        results = {}
        
        # Training evaluation
        if train_envs:
            train_results = self._run_envs(train_envs, policy, "train")
            results.update(train_results)
        
        # Testing evaluation
        if test_envs:
            test_results = self._run_envs(test_envs, policy, "test")
            results.update(test_results)

        # Close environments
        for env in train_envs + test_envs:
            try:
                env.close()
            except:
                pass

        return results

    def _run_envs(self, envs, policy, prefix):
        """Run evaluation on a list of environments"""
        device = policy.device
        dtype = policy.dtype
        
        results = []
        video_paths = []
        
        for env_idx, env in enumerate(envs):
            try:
                # Reset environment
                obs = env.reset()
                
                # Initialize past_action buffer - Fix past_action shape to store n_obs_steps actions
                past_action = np.zeros((self.n_obs_steps, 10))  # 10-dimensional action (4 robot + 6 human)
                
                episode_reward = 0
                episode_success = False
                step_count = 0
                
                while step_count < self.max_steps:
                    # create obs dict - Observation data is read directly from dataset without modification
                    np_obs_dict = {"obs": obs.astype(np.float32)}
                    if self.past_action:
                        np_obs_dict["past_action"] = past_action.astype(np.float32)
                    
                    # device transfer
                    obs_dict = dict_apply(
                        np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                    )
                    
                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)
                    
                    # device_transfer
                    np_action_dict = dict_apply(
                        action_dict, lambda x: x.detach().to("cpu").numpy()
                    )
                    
                    action = np_action_dict["action"]
                    
                    # step env
                    obs, reward, done, info = env.step(action)
                    
                    # update past action
                    past_action[:-1] = past_action[1:]
                    # Take first action step, since action shape is (n_action_steps, action_dim)
                    past_action[-1] = action[0]
                    
                    episode_reward += reward
                    step_count += 1
                    
                    if done:
                        episode_success = True
                        break
                
                results.append({
                    'reward': episode_reward,
                    'success': episode_success,
                    'steps': step_count
                })
                
            except Exception as e:
                print(f"Error running environment {env_idx}: {e}")
                results.append({
                    'reward': 0.0,
                    'success': False,
                    'steps': 0
                })
        
        # Calculate statistics
        rewards = [r['reward'] for r in results]
        successes = [r['success'] for r in results]
        
        return {
            f'{prefix}_mean_score': np.mean(rewards),
            f'{prefix}_mean_success': np.mean(successes),
            f'{prefix}_std_score': np.std(rewards),
            f'{prefix}_std_success': np.std(successes),
        }

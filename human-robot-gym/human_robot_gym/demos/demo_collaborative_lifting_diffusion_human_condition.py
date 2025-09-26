"""Demo script for the collaborative lifting environment using Diffusion model with human action conditioning.
Uses a trained Diffusion model to control the robot with human action as condition.

Diffusion模型输出说明:
- 输入: 23维观察状态 + 6维人类动作历史 (obs_keys + past_action)
- 输出: 10维动作向量，其中前4维是机器人动作:
  [x_delta, y_delta, z_delta, gripper_action, human_left_x, human_left_y, human_left_z, human_right_x, human_right_y, human_right_z]
- 机器人动作维度:
  - x_delta: x方向位置增量
  - y_delta: y方向位置增量  
  - z_delta: z方向位置增量
  - gripper_action: 夹爪动作 (0=开, 1=关)

Pressing 'o' switches between Diffusion policy and keyboard control.

Author:
    Modified from demo_collaborative_lifting_diffusion.py to use human action conditioning
"""
import robosuite as suite
import time
import numpy as np
import glfw
import torch
import os
import sys
import hydra
import dill
from omegaconf import OmegaConf

from robosuite.controllers import load_controller_config

from human_robot_gym.utils.mjcf_utils import file_path_completion, merge_configs
from human_robot_gym.utils.cart_keyboard_controller import KeyboardControllerAgentCart
from human_robot_gym.utils.env_util import ExpertObsWrapper
import human_robot_gym.robots  # noqa: F401
from human_robot_gym.wrappers.visualization_wrapper import VisualizationWrapper
from human_robot_gym.wrappers.collision_prevention_wrapper import (
    CollisionPreventionWrapper,
)
from human_robot_gym.wrappers.ik_position_delta_wrapper import IKPositionDeltaWrapper

# Add diffusion_copolicy to path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.join(current_dir, '..', '..', '..')
diffusion_copolicy_path = os.path.join(workspace_root, 'diffusion_copolicy')
sys.path.append(diffusion_copolicy_path)
sys.path.append(os.path.join(diffusion_copolicy_path, 'diffusion_policy'))

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy

class DiffusionHumanConditionAgent:
    """Diffusion agent for collaborative lifting task with human action conditioning"""
    
    def __init__(self, model_path, device="gpu", robot_action_dim=4, human_action_dim=6):
        self.device = device
        self.robot_action_dim = robot_action_dim
        self.human_action_dim = human_action_dim
        
        # Load checkpoint
        print(f"Loading Diffusion model with human conditioning from: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", pickle_module=dill)
        cfg = checkpoint["cfg"]
        
        # Manually calculate configuration values to avoid eval interpolation issues
        # Extract basic parameters from configuration
        obs_dim = cfg.obs_dim  # 23 (base observation)
        action_dim = cfg.action_dim  # 10
        obs_as_cond = cfg.obs_as_cond  # True
        human_act_as_cond = cfg.human_act_as_cond  # True
        human_action_dim = cfg.human_action_dim  # 6

        # Manually calculate model dimensions
        if obs_as_cond:
            input_dim = action_dim  # 10
            # For human condition model, cond_dim includes human action: obs_dim + human_action_dim
            cond_dim = obs_dim + human_action_dim  # 23 + 6 = 29
        else:
            input_dim = obs_dim + action_dim  # 23 + 10 = 33
            cond_dim = 0

        output_dim = input_dim  # 10

        print(f"✓ Manually calculated configuration values:")
        print(f"  - obs_dim: {obs_dim} (base observation)")
        print(f"  - human_action_dim: {human_action_dim}")
        print(f"  - action_dim: {action_dim}")
        print(f"  - obs_as_cond: {obs_as_cond}")
        print(f"  - human_act_as_cond: {human_act_as_cond}")
        print(f"  - input_dim: {input_dim}")
        print(f"  - output_dim: {output_dim}")
        print(f"  - cond_dim: {cond_dim} (obs + human_action)")

        # Manually create model instance to avoid eval interpolation issues
        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=cfg.horizon,
            n_obs_steps=cfg.n_obs_steps,
            cond_dim=cond_dim,
            n_layer=cfg.policy.model.n_layer,
            n_head=cfg.policy.model.n_head,
            n_emb=cfg.policy.model.n_emb,
            p_drop_emb=cfg.policy.model.p_drop_emb,
            p_drop_attn=cfg.policy.model.p_drop_attn,
            causal_attn=cfg.policy.model.causal_attn,
            time_as_cond=cfg.policy.model.time_as_cond,
            obs_as_cond=obs_as_cond,
            human_act_as_cond=human_act_as_cond,
            n_cond_layers=cfg.policy.model.n_cond_layers,
        )
        
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
            
            # Filter out keys that don't belong to the model, but keep normalizer-related keys
            print("✓ Filtering model state dict...")
            filtered_state_dict = {}
            normalizer_state_dict = {}
            for key, value in model_state_dict.items():
                # Skip mask_generator related keys, but keep normalizer related keys
                if key.startswith("mask_generator"):
                    print(f"  - Skipping key: {key}")
                elif key.startswith("normalizer"):
                    # Save normalizer related keys to normalizer_state_dict
                    normalizer_key = key.replace("normalizer.", "")
                    normalizer_state_dict[normalizer_key] = value
                else:
                    filtered_state_dict[key] = value
            
            print(f"✓ Loaded {len(filtered_state_dict)} model parameters")
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # If normalizer not found in state_dicts, try to get it from elsewhere
            if not normalizer_state_dict:
                normalizer_state_dict = checkpoint["state_dicts"].get("normalizer")
        else:
            raise ValueError("Unknown checkpoint format")
        
        # Manually create noise scheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.policy.noise_scheduler.num_train_timesteps,
            beta_start=cfg.policy.noise_scheduler.beta_start,
            beta_end=cfg.policy.noise_scheduler.beta_end,
            beta_schedule=cfg.policy.noise_scheduler.beta_schedule,
            variance_type=cfg.policy.noise_scheduler.variance_type,
            clip_sample=cfg.policy.noise_scheduler.clip_sample,
            prediction_type=cfg.policy.noise_scheduler.prediction_type,
        )

        # Manually create policy
        self.policy = DiffusionTransformerLowdimPolicy(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=cfg.horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=cfg.n_action_steps,
            n_obs_steps=cfg.n_obs_steps,
            num_inference_steps=cfg.policy.num_inference_steps,
            obs_as_cond=obs_as_cond,
            human_act_as_cond=human_act_as_cond,
            pred_action_steps_only=cfg.pred_action_steps_only,
            robot_action_dim=cfg.robot_action_dim,
            human_action_dim=cfg.human_action_dim,
        )
        
        # Load normalizer (if exists)
        if normalizer_state_dict is not None:
            self.policy.normalizer.load_state_dict(normalizer_state_dict)
            print("✓ Successfully loaded normalizer")
        else:
            print("⚠ Normalizer not found, using default initialization")
        
        # Move model to device
        self.policy.eval().to(device)
        print(f"✓ Model moved to device: {device}")
        
        # Set inference parameters
        self.policy.num_inference_steps = cfg.policy.get("num_inference_steps", 100)
        
        # Initialize observation buffer for multi-step prediction
        self.obs_buffer = None
        self.past_action_buffer = None
        self.n_obs_steps = cfg.n_obs_steps
        
        print("✓ Diffusion model with human conditioning loaded successfully")
    
    def reset(self):
        """Reset the agent state"""
        self.policy.reset()
        self.obs_buffer = None
        self.past_action_buffer = None
    
    def predict_action(self, observation, human_action=None):
        """
        Predict action from observation and human action
        Args:
            observation: 23-dimensional observation array (base observation)
            human_action: 6-dimensional human action array (required for human condition model)
        Returns:
            action: 4-dimensional robot action (first robot_action_dim dimensions)
        """
        # Convert observation to tensor
        if not isinstance(observation, torch.Tensor):
            obs_tensor = torch.from_numpy(observation).float()
        else:
            obs_tensor = observation
        
        # Convert human action to tensor
        if human_action is None:
            raise ValueError("human_action is required for human condition model")
        if not isinstance(human_action, torch.Tensor):
            human_action_tensor = torch.from_numpy(human_action).float()
        else:
            human_action_tensor = human_action
        
        # Update observation buffer with base observation only (23D)
        # Normalizer only processes base observation, human action is combined later
        if self.obs_buffer is None:
            # Initialize buffer with current base observation
            self.obs_buffer = obs_tensor.unsqueeze(0).repeat(self.n_obs_steps, 1)
        else:
            # Shift buffer and add new base observation
            self.obs_buffer = torch.cat([self.obs_buffer[1:], obs_tensor.unsqueeze(0)], dim=0)
        
        # Prepare input dictionary with base observation only (23D)
        obs_dict = {
            "obs": self.obs_buffer.unsqueeze(0).to(self.device)  # Add batch dimension
        }
        
        # Add past_action for human condition
        if self.past_action_buffer is None:
            # Initialize with zeros
            self.past_action_buffer = torch.zeros(self.n_obs_steps, self.robot_action_dim + self.human_action_dim).to(self.device)
        
        # Create full action vector: [robot_action(4D), human_action(6D)]
        # For past_action, we use zeros for robot action and actual human action
        full_past_action = torch.cat([
            torch.zeros(self.robot_action_dim).to(self.device),  # Robot action (will be filled by model prediction)
            human_action_tensor[:self.human_action_dim].to(self.device)  # Human action
        ])
        
        # Shift buffer and add new action
        self.past_action_buffer = torch.cat([self.past_action_buffer[1:], full_past_action.unsqueeze(0)], dim=0)
        
        obs_dict["past_action"] = self.past_action_buffer.unsqueeze(0)
        
        # Predict action
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            full_action = result["action"][0]  # Remove batch dimension
        
        # Extract robot action (first robot_action_dim dimensions)
        # Diffusion输出10维动作，前4维是机器人动作:
        # [x_delta, y_delta, z_delta, gripper_action, human_left_x, human_left_y, human_left_z, human_right_x, human_right_y, human_right_z]
        robot_action = full_action[0, :self.robot_action_dim]  # 取第一个时间步的前4维作为机器人动作
        
        # Update past_action buffer with predicted action for next step
        predicted_full_action = full_action[0]
        self.past_action_buffer = torch.cat([self.past_action_buffer[1:], predicted_full_action.unsqueeze(0)], dim=0)
        
        return robot_action.cpu().numpy()

if __name__ == "__main__":
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
        has_renderer=True,  # make sure we can render to the screen
        render_camera=None,
        render_collision_mesh=False,
        control_freq=10,  # control should happen fast enough so that simulation looks smooth
        hard_reset=False,
        horizon=1000,
        done_at_success=False,
        controller_configs=controller_configs,
        shield_type="SSM",  # Shield mode, can be "SSM" or "PFL"
        visualize_failsafe_controller=False,
        visualize_pinocchio=False,
        base_human_pos_offset=[0.0, 0.0, 0.0],
        human_rand=[0, 0.0, 0.0],
        verbose=True,
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
    env = VisualizationWrapper(env)
    action_limits = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
    env = IKPositionDeltaWrapper(env=env, urdf_file=pybullet_urdf_file, action_limits=action_limits)
    kb_agent = KeyboardControllerAgentCart(env=env)

    use_kb_agent = False

    def switch_agent():
        global use_kb_agent
        use_kb_agent = not use_kb_agent
        if use_kb_agent:
            print("Switched to keyboard control")
        else:
            print("Switched to Diffusion model with human conditioning control")

    def toggle_board():
        if rsenv.human_holds_board:
            rsenv.human_drop_board()
        else:
            rsenv.human_pickup_board()

    kb_agent.add_keypress_callback(glfw.KEY_O, lambda *_: switch_agent())
    kb_agent.add_keypress_callback(glfw.KEY_B, lambda *_: toggle_board())

    # Load Diffusion model with human conditioning
    model_path = os.path.join(workspace_root, "diffusion_copolicy", "data", "outputs", "diffusion_model_lifting_human_cond", "best_model.ckpt")
    
    # Check if best_model.ckpt exists, otherwise try to find the latest checkpoint
    if not os.path.exists(model_path):
        checkpoints_dir = os.path.join(workspace_root, "diffusion_copolicy", "data", "outputs", "diffusion_model_lifting_human_cond")
        if os.path.exists(checkpoints_dir):
            # Look for checkpoint files in subdirectories
            checkpoint_files = []
            for root, dirs, files in os.walk(checkpoints_dir):
                for file in files:
                    if file.endswith(('.ckpt', '.pt')):
                        checkpoint_files.append(os.path.join(root, file))
            
            if checkpoint_files:
                # Sort by modification time and take the latest
                checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                model_path = checkpoint_files[0]
                print(f"Using latest checkpoint: {os.path.basename(model_path)}")
            else:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
        else:
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Initialize Diffusion agent with human conditioning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    diffusion_agent = DiffusionHumanConditionAgent(
        model_path=model_path,
        device=device,
        robot_action_dim=4,  # Robot action dimension for collaborative lifting
        human_action_dim=6   # Human action dimension for collaborative lifting
    )

    from scipy.spatial.transform import Rotation

    # Planning time statistics variables
    planning_times = []
    episode_planning_times = []
    
    for i_episode in range(20):
        observation = env.reset()
        diffusion_agent.reset()  # Reset agent state for new episode

        t1 = time.time()
        t = 0
        episode_planning_times = []  # Reset timing statistics for each episode
        while True:
            t += 1
            
            if use_kb_agent:
                action = kb_agent()
            else:
                # Use Diffusion model with human conditioning
                # Extract 23-dimensional observation from the environment
                # The observation should contain the obs_keys data (23 dimensions)
                
                # Get the raw observation from the environment
                raw_obs = env.unwrapped._get_observations()
                
                # Extract the 23-dimensional observation from obs_keys
                # Based on the SAC dataset structure:
                # obs_keys: 23维度
                #   - robot0_eef_pos        # 机器人末端执行器位置 (3维)
                #   - robot0_gripper_qpos   # 夹爪两个关节的位置 (2维)
                #   - robot0_gripper_qvel   # 夹爪两个关节的速度 (2维)
                #   - vec_eef_to_human_head # 到人类头部的向量 (3维)
                #   - vec_eef_to_human_lh   # 到人类左手的向量 (3维)
                #   - vec_eef_to_human_rh   # 到人类右手的向量 (3维)
                #   - board_quat            # 木板方向四元数 (4维)
                #   - board_balance         # 木板平衡度 (1维)
                #   - board_gripped         # 木板是否被夹持 (1维)
                #   - dist_eef_to_human_head # 到人类头部的距离 (1维)
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
                
                obs_23d = np.array(obs_23d, dtype=np.float32)
                
                if len(obs_23d) != 23:
                    print(f"Warning: Expected 23-dimensional observation, got {len(obs_23d)} dimensions")
                    # Pad or truncate as needed
                    if len(obs_23d) < 23:
                        obs_23d = np.pad(obs_23d, (0, 23 - len(obs_23d)), 'constant')
                    else:
                        obs_23d = obs_23d[:23]
                
                # Generate human action (simulated for demo purposes)
                # In real scenario, this would come from human input or another model
                # For now, we'll use a simple pattern based on time
                human_action = np.array([
                    np.sin(t * 0.1) * 0.1,  # left hand x
                    np.cos(t * 0.1) * 0.1,  # left hand y
                    np.sin(t * 0.05) * 0.05,  # left hand z
                    np.cos(t * 0.1) * 0.1,  # right hand x
                    np.sin(t * 0.1) * 0.1,  # right hand y
                    np.cos(t * 0.05) * 0.05,  # right hand z
                ], dtype=np.float32)
                
                # Get action from Diffusion model with human conditioning
                planning_start = time.time()
                robot_action = diffusion_agent.predict_action(obs_23d, human_action)
                planning_end = time.time()
                
                # Record planning time
                planning_time = (planning_end - planning_start) * 1000  # Convert to milliseconds
                planning_times.append(planning_time)
                episode_planning_times.append(planning_time)
                
                # Diffusion输出的robot action维度说明:
                # action = [
                #     x_delta,      # 维度 0: x方向位置增量
                #     y_delta,      # 维度 1: y方向位置增量  
                #     z_delta,      # 维度 2: z方向位置增量
                #     gripper_action # 维度 3: 夹爪动作 (0=开, 1=关)
                # ]
                action = robot_action  # 直接使用4维机器人动作

            env.viewer.viewer.add_marker(
                pos=env.sim.data.get_site_xpos("gripper0_grip_site"),
                type=100,
                size=[0.005, 0.005, np.linalg.norm(action[:3]) * 5],
                mat=Rotation.align_vectors(
                    action[:3].reshape(1, -1),
                    np.array([0, 0, 0.1]).reshape(1, -1)
                )[0].as_matrix(),
                rgba=[1, 0, 0, 1],
                label="",
                shininess=0.0,
            )
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        # Calculate and output timing statistics for each episode
        episode_avg_planning = np.mean(episode_planning_times) if episode_planning_times else 0
        episode_max_planning = np.max(episode_planning_times) if episode_planning_times else 0
        episode_min_planning = np.min(episode_planning_times) if episode_planning_times else 0
        
        print("Episode {}, fps = {}".format(i_episode, t / (time.time() - t1)))
        print("  📊 Planning Time Stats (ms): Avg={:.2f}, Max={:.2f}, Min={:.2f}".format(
            episode_avg_planning, episode_max_planning, episode_min_planning))
    
    # Final statistics output
    if planning_times:
        overall_avg = np.mean(planning_times)
        overall_max = np.max(planning_times)
        overall_min = np.min(planning_times)
        overall_std = np.std(planning_times)
        
        print("\n" + "="*60)
        print("🎯 Diffusion Human Condition Model Planning Time Statistics Summary")
        print("="*60)
        print(f"📈 Total Planning Count: {len(planning_times)}")
        print(f"⏱️  Average Planning Time: {overall_avg:.2f} ms")
        print(f"🚀 Fastest Planning Time: {overall_min:.2f} ms")
        print(f"🐌 Slowest Planning Time: {overall_max:.2f} ms")
        print(f"📊 Standard Deviation: {overall_std:.2f} ms")
        print(f"🎯 Real-time Performance: {'✅ Excellent' if overall_avg < 50 else '⚠️  Needs Optimization' if overall_avg < 100 else '❌ Poor Performance'}")
        print("="*60)

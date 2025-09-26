# Flow Matching Collaborative Lifting Demo Summary

## Overview
Successfully created a collaborative lifting demo using Flow Matching model, replacing the original expert policy.

## File Location
`human_robot_gym/demos/demo_collaborative_lifting_flow_matching.py`

## Flow Matching Model Input/Output

### Input: 23-dimensional Observation State (obs_keys)
```
obs_keys: 23 dimensions
  - robot0_eef_pos        # Robot end-effector position (3D)
  - robot0_gripper_qpos   # Gripper joint positions (2D)
  - robot0_gripper_qvel   # Gripper joint velocities (2D)
  - vec_eef_to_human_head # Vector to human head (3D)
  - vec_eef_to_human_lh   # Vector to human left hand (3D)
  - vec_eef_to_human_rh   # Vector to human right hand (3D)
  - board_quat            # Board orientation quaternion (4D)
  - board_balance         # Board balance (1D)
  - board_gripped         # Board gripped status (1D)
  - dist_eef_to_human_head # Distance to human head (1D)
```

### Output: 10-dimensional Action Vector
Flow Matching model outputs a 10-dimensional action vector with the following structure:
```
[x_delta, y_delta, z_delta, gripper_action, human_left_x, human_left_y, human_left_z, human_right_x, human_right_y, human_right_z]
```

### Robot Action (First 4 Dimensions)
Extract the first 4 dimensions from the 10-dimensional output as robot action:
```python
action = [
    x_delta,      # Dimension 0: x-direction position delta
    y_delta,      # Dimension 1: y-direction position delta  
    z_delta,      # Dimension 2: z-direction position delta
    gripper_action # Dimension 3: gripper action (0=open, 1=close)
]
```

## Main Features

### 1. Automatic Model Loading
- Prioritizes using `best_model.ckpt`
- Automatically finds the latest checkpoint if not available
- Supports diffusion_policy and flow_policy format conversion

### 2. Observation Data Processing
- Extracts 23-dimensional obs_keys data from raw environment observations
- Automatically calculates missing distance information
- Handles observation dimension mismatch cases

### 3. Action Output Processing
- Extracts first 4 dimensions from Flow Matching's 10-dimensional output
- Supports multi-step prediction and observation buffering
- Real-time past_action buffer updates

### 4. Interactive Control
- Press 'O' key: Switch between Flow Matching control and keyboard control
- Press 'B' key: Toggle board gripping status
- Real-time action vector visualization

## Technical Features

### Device Support
- Automatically detects and uses CUDA (if available)
- Supports CPU fallback

### Error Handling
- Complete checkpoint loading error handling
- Automatic observation dimension mismatch repair
- Model format compatibility processing

### Performance Optimization
- Uses torch.no_grad() for inference
- Observation buffering to avoid redundant calculations
- Multi-step prediction support

## Usage Instructions

1. Ensure Flow Matching model training is completed
2. Run the demo:
   ```bash
   cd /home/ubuntu/IRoMan/human-robot-gym
   python human_robot_gym/demos/demo_collaborative_lifting_flow_matching.py
   ```
3. Use keyboard controls:
   - 'O' key: Switch control mode
   - 'B' key: Toggle board status

## Model Path
Default model path:
```
/home/ubuntu/IRoMan/flow_copolicy/outputs/flow_matching_collaborative_lifting/checkpoints/best_model.ckpt
```

If best_model.ckpt doesn't exist, it will automatically use the latest checkpoint file.

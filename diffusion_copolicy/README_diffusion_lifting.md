# Diffusion Co-Policy for Collaborative Lifting Task

This project implements Diffusion Co-Policy training scripts for collaborative lifting tasks, based on the structure of `train_flow_matching_lifting.py`, but using Diffusion Model algorithm instead of Flow Matching.

## Main Differences

### Diffusion vs Flow Matching

1. **Prediction Target**:
   - **Diffusion**: Predict noise (epsilon)
   - **Flow Matching**: Predict velocity field

2. **Loss Function**:
   - **Diffusion**: MSE(predicted noise, true noise)
   - **Flow Matching**: MSE(predicted velocity field, true velocity field)

3. **Inference Process**:
   - **Diffusion**: DDPM sampler step-by-step denoising
   - **Flow Matching**: ODE solver integration

4. **Network Structure**: Same (Transformer)

## File Structure

```
diffusion_copolicy/
├── train_diffusion_lifting.py                    # Training script without human action as condition
├── train_diffusion_lifting_human_cond.py         # Training script with human action as condition
├── diffusion_policy/
│   ├── config/
│   │   ├── task/
│   │   │   └── lifting_lowdim.yaml              # Task configuration file
│   │   ├── train_diffusion_transformer_lowdim_lifting.yaml           # Training config (no human cond)
│   │   └── train_diffusion_transformer_lowdim_lifting_human_cond.yaml # Training config (with human cond)
│   ├── dataset/
│   │   └── lifting_lowdim_dataset.py            # Dataset class
│   └── env_runner/
│       └── lifting_lowdim_runner.py             # Environment runner
└── README_diffusion_lifting.md                   # This file
```

## Data Format

### Observation State (23-dimensional)
- `robot0_eef_pos`: (3D) Robot end-effector position
- `robot0_gripper_qpos`: (2D) Gripper joint position
- `robot0_gripper_qvel`: (2D) Gripper joint velocity
- `vec_eef_to_human_head`: (3D) Vector to human head
- `vec_eef_to_human_lh`: (3D) Vector to human left hand
- `vec_eef_to_human_rh`: (3D) Vector to human right hand
- `board_quat`: (4D) Board orientation quaternion
- `board_balance`: (1D) Board balance
- `board_gripped`: (1D) Whether board is gripped
- `dist_eef_to_human_head`: (1D) Distance to human head

### Action State (10-dimensional)
- **Robot Action** (4D): `[x_delta, y_delta, z_delta, gripper_action]`
- **Human Action** (6D): `[human_left_x, human_left_y, human_left_z, human_right_x, human_right_y, human_right_z]`

## Usage

### 1. Without Human Action as Condition

```bash
cd /home/ubuntu/IRoMan/diffusion_copolicy
python train_diffusion_lifting.py
```

### 2. With Human Action as Condition

```bash
cd /home/ubuntu/IRoMan/diffusion_copolicy
python train_diffusion_lifting_human_cond.py
```

## Configuration Parameters

### Main Parameters
- `obs_dim`: 23 (observation dimension)
- `action_dim`: 10 (action dimension)
- `robot_action_dim`: 4 (robot action dimension)
- `human_action_dim`: 6 (human action dimension)
- `horizon`: 8 (prediction horizon)
- `n_obs_steps`: 2 (observation steps)
- `n_action_steps`: 6 (action steps)

### Model Parameters
- `n_layer`: 6 (Transformer layers)
- `n_head`: 4 (attention heads)
- `n_emb`: 128 (embedding dimension)
- `num_inference_steps`: 100 (inference steps)

### Training Parameters
- `batch_size`: 32
- `learning_rate`: 2.0e-4
- `num_epochs`: 500
- `device`: "cuda:0"

## Output

After training completion, models and logs will be saved in the following directories:
- Without human condition: `outputs/diffusion_lifting/`
- With human condition: `outputs/diffusion_lifting_human_cond/`

## Dependencies

- PyTorch
- Hydra
- OmegaConf
- Diffusers
- human-robot-gym (for environment)
- robosuite (for environment)

## Notes

1. Ensure dataset file `data/table/collaborative_lifting_sac_350.zarr` exists
2. Ensure human-robot-gym environment is properly installed
3. Adjust batch_size according to GPU memory
4. Model architecture parameters can be adjusted as needed

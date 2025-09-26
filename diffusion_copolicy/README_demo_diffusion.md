# Diffusion Co-policy Demo Script

## Overview

`demo_collaborative_lifting_diffusion.py` is a demo script for controlling robots using trained Diffusion Co-policy models in collaborative lifting environments.

## Features

- **Diffusion Model Control**: Use trained Diffusion Transformer models for robot control
- **Keyboard Control Toggle**: Press 'O' key to switch between Diffusion model control and keyboard control
- **Board Control**: Press 'B' key to control whether human holds the board
- **Real-time Visualization**: Display robot action direction and magnitude

## Model Requirements

### Supported Models
- **Model Type**: Diffusion Transformer Co-policy
- **Task**: Collaborative Lifting
- **Condition**: Without human action as condition (`human_act_as_cond=False`)

### Model Path
The script will automatically search for models at the following path:
```
diffusion_copolicy/data/outputs/diffusion_model_lifting_no_human_cond/best_model.ckpt
```

If `best_model.ckpt` does not exist, the script will automatically find the latest checkpoint file.

## Usage

### 1. Environment Setup
Ensure all dependencies are installed:
```bash
# Activate conda environment
conda activate robodiff

# Ensure human-robot-gym is installed
cd /home/ubuntu/IRoMan/human-robot-gym
pip install -e .
```

### 2. Run Demo
```bash
cd /home/ubuntu/IRoMan/diffusion_copolicy
python demo_collaborative_lifting_diffusion.py
```

### 3. Control Instructions
- **'O' key**: Switch between Diffusion model control and keyboard control
- **'B' key**: Control whether human holds the board
- **Keyboard Control**: Use WASD keys to control robot movement

## Model Input/Output

### Input (23-dimensional observation)
```
robot0_eef_pos        # Robot end-effector position (3D)
robot0_gripper_qpos   # Gripper joint position (2D)
robot0_gripper_qvel   # Gripper joint velocity (2D)
vec_eef_to_human_head # Vector to human head (3D)
vec_eef_to_human_lh   # Vector to human left hand (3D)
vec_eef_to_human_rh   # Vector to human right hand (3D)
board_quat            # Board orientation quaternion (4D)
board_balance         # Board balance (1D)
board_gripped         # Whether board is gripped (1D)
dist_eef_to_human_head # Distance to human head (1D)
```

### Output (4-dimensional robot action)
```
x_delta      # x-direction position delta
y_delta      # y-direction position delta  
z_delta      # z-direction position delta
gripper_action # Gripper action (0=open, 1=close)
```

## Technical Details

### Diffusion vs Flow Matching
- **Diffusion**: Use DDPM sampler for step-by-step denoising, 100 inference steps
- **Flow Matching**: Use ODE solver integration, 30 inference steps
- **Network Structure**: Same Transformer architecture

### Observation History
- **Observation Steps**: 2 steps (`n_obs_steps=2`)
- **Prediction Length**: 8 steps (`horizon=8`)
- **Action Steps**: 6 steps (`n_action_steps=6`)

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```
   FileNotFoundError: No checkpoint files found
   ```
   **Solution**: Ensure model training is completed, check if model path is correct

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU or reduce inference steps

3. **Observation Dimension Mismatch**
   ```
   Warning: Expected 23-dimensional observation, got X dimensions
   ```
   **Solution**: Check environment configuration, ensure observation space is correct

### Debug Mode
If encountering issues, you can add debug information:
```python
# Add to script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### GPU Acceleration
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Inference Steps Adjustment
```python
# Reduce inference steps for faster speed
self.policy.num_inference_steps = 50  # Default 100
```

## Extended Features

### Support for Human Action as Condition
To use models with `human_act_as_cond=True`, you need to:
1. Modify model path to point to corresponding checkpoint
2. Ensure `past_action` data is available
3. Update observation buffer logic

### Multi-Model Comparison
You can load multiple models simultaneously for comparison:
```python
diffusion_agent = DiffusionAgent(model_path_diffusion, device)
flow_matching_agent = FlowMatchingAgent(model_path_flow, device)
```

## Related Files

- `train_diffusion_lifting.py`: Training script
- `train_diffusion_lifting_human_cond.py`: Training script with human condition
- `diffusion_policy/config/`: Configuration file directory
- `diffusion_policy/dataset/lifting_lowdim_dataset.py`: Dataset class
- `diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`: Policy class

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Diffusion Transformer Network Architecture            │
│                              (Table-Carrying Task)                          │
└─────────────────────────────────────────────────────────────────────────────┘

Input Data Flow:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Observation │    │   Action    │    │    Time     │
│   (18D)     │    │   (4D)      │    │   (1D)      │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Input Embedding Layer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Observation Embedding: Linear(18 → 256)                                 │
│  • Action Embedding: Linear(4 → 256)                                       │
│  • Time Embedding: SinusoidalPosEmb(256)                                   │
│  • Positional Embedding: nn.Parameter(1, 12, 256)                         │
│  • Dropout: 0.0                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Transformer Encoder (8 Layers)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1-8:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Multi-Head Self-Attention (4 heads, 256 dim)                          │ │
│  │  ├─ Query: Linear(256 → 256)                                           │ │
│  │  ├─ Key: Linear(256 → 256)                                             │ │
│  │  ├─ Value: Linear(256 → 256)                                           │ │
│  │  └─ Output: Linear(256 → 256)                                          │ │
│  │                                                                         │ │
│  │  Layer Normalization + Residual Connection                              │ │
│  │                                                                         │ │
│  │  Feed Forward Network:                                                  │ │
│  │  ├─ Linear(256 → 1024)                                                 │ │
│  │  ├─ GELU Activation                                                    │ │
│  │  ├─ Linear(1024 → 256)                                                 │ │
│  │  └─ Dropout(0.3)                                                       │ │
│  │                                                                         │ │
│  │  Layer Normalization + Residual Connection                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Output Layer                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Linear Layer: Linear(256 → 4)                                           │
│  • Output Dimension: 4 (Action Dimension)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Noise Prediction                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Prediction Target: Noise ε (epsilon)                                    │
│  • Loss Function: MSE(Predicted Noise, True Noise)                         │
│  • Inference Process: DDPM Denoising Sampling (100 steps)                  │
└─────────────────────────────────────────────────────────────────────────────┘


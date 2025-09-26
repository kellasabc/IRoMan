# Diffusion Co-policy Demo Scripts

## Overview

This directory contains two demo scripts using Diffusion Co-policy models:

1. **`demo_collaborative_lifting_diffusion.py`** - Version without human action as condition
2. **`demo_collaborative_lifting_diffusion_human_condition.py`** - Version with human action as condition

## Features

- **Diffusion Model Control**: Uses trained Diffusion Transformer model for robot control
- **Keyboard Control Toggle**: Press 'O' key to switch between Diffusion model control and keyboard control
- **Board Control**: Press 'B' key to control whether human holds the board
- **Real-time Visualization**: Shows robot action direction and magnitude
- **Human Action Condition**: Supports using human action history as model input condition

## Model Requirements

### Supported Model Types
- **Model Type**: Diffusion Transformer Co-policy
- **Task**: Collaborative Lifting
- **Condition Types**: 
  - Version 1: Without human action as condition (`human_act_as_cond=False`)
  - Version 2: With human action as condition (`human_act_as_cond=True`)

### Model Paths
Scripts will automatically search for models at the following paths:

#### Version 1 (No Human Condition):
```
diffusion_copolicy/data/outputs/diffusion_model_lifting_no_human_cond/best_model.ckpt
```

#### Version 2 (With Human Condition):
```
diffusion_copolicy/data/outputs/diffusion_model_lifting_human_cond/best_model.ckpt
```

If `best_model.ckpt` does not exist, scripts will automatically find the latest checkpoint file.

## Usage

### 1. Environment Setup
Ensure all dependencies are installed:
```bash
# Activate conda environment
conda activate robodiff

# Ensure human-robot-gym is installed
cd /home/ubuntu/IRoMan/human-robot-gym
pip install -e .

# Ensure diffusion_copolicy is installed
cd /home/ubuntu/IRoMan/diffusion_copolicy
pip install -e .
```

### 2. Run Demo

#### Version 1 (No Human Condition):
```bash
cd /home/ubuntu/IRoMan/human-robot-gym/human_robot_gym/demos
python demo_collaborative_lifting_diffusion.py
```

#### Version 2 (With Human Condition):
```bash
cd /home/ubuntu/IRoMan/human-robot-gym/human_robot_gym/demos
python demo_collaborative_lifting_diffusion_human_condition.py
```

### 3. Control Instructions
- **'O' Key**: Switch between Diffusion model control and keyboard control
- **'B' Key**: Control whether human holds the board
- **Keyboard Control**: Use WASD keys to control robot movement

## Model Input/Output

### Input (23-dimensional observation)
```
robot0_eef_pos        # Robot end-effector position (3D)
robot0_gripper_qpos   # Gripper joint positions (2D)
robot0_gripper_qvel   # Gripper joint velocities (2D)
vec_eef_to_human_head # Vector to human head (3D)
vec_eef_to_human_lh   # Vector to human left hand (3D)
vec_eef_to_human_rh   # Vector to human right hand (3D)
board_quat            # Board quaternion orientation (4D)
board_balance         # Board balance (1D)
board_gripped         # Whether board is gripped (1D)
dist_eef_to_human_head # Distance to human head (1D)
```

### Input (Human Action Condition - Version 2 only)
```
past_action           # Human action history (6D × n_obs_steps)
                      # [human_left_x, human_left_y, human_left_z, 
                      #  human_right_x, human_right_y, human_right_z]
```

### 输出 (4维机器人动作)
```
x_delta      # x方向位置增量
y_delta      # y方向位置增量  
z_delta      # z方向位置增量
gripper_action # 夹爪动作 (0=开, 1=关)
```

## 技术细节

### Diffusion vs Flow Matching
- **Diffusion**: 使用DDPM采样器逐步去噪，推理步数100步
- **Flow Matching**: 使用ODE求解器积分，推理步数30步
- **网络结构**: 相同的Transformer架构

### 观察历史
- **观察步数**: 2步 (`n_obs_steps=2`)
- **预测长度**: 8步 (`horizon=8`)
- **动作步数**: 6步 (`n_action_steps=6`)

### 人类动作模拟 (版本2)
在演示中，人类动作使用简单的正弦波模式模拟：
```python
human_action = np.array([
    0.01 * np.sin(t * 0.1),  # human_left_x
    0.01 * np.cos(t * 0.1),  # human_left_y
    0.005 * np.sin(t * 0.15), # human_left_z
    0.01 * np.cos(t * 0.12),  # human_right_x
    0.01 * np.sin(t * 0.08),  # human_right_y
    0.005 * np.cos(t * 0.18), # human_right_z
], dtype=np.float32)
```

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   FileNotFoundError: No checkpoint files found
   ```
   **解决方案**: 确保模型已训练完成，检查模型路径是否正确

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案**: 使用CPU运行或减少推理步数

3. **观察维度不匹配**
   ```
   Warning: Expected 23-dimensional observation, got X dimensions
   ```
   **解决方案**: 检查环境配置，确保观察空间正确

4. **人类动作条件错误**
   ```
   KeyError: 'past_action'
   ```
   **解决方案**: 确保使用正确的模型版本和配置

### 调试模式
如果遇到问题，可以添加调试信息：
```python
# 在脚本中添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

### GPU加速
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 推理步数调整
```python
# 减少推理步数以提高速度
self.policy.num_inference_steps = 50  # 默认100
```

## 扩展功能

### 真实人类动作输入
要使用真实的人类动作数据，需要：
1. 连接人类动作捕捉系统
2. 修改 `human_action` 生成逻辑
3. 确保动作数据格式正确 (6维向量)

### 多模型比较
可以同时加载多个模型进行比较：
```python
diffusion_agent = DiffusionAgent(model_path_diffusion, device)
flow_matching_agent = FlowMatchingAgent(model_path_flow, device)
```

## 相关文件

### 训练脚本
- `diffusion_copolicy/train_diffusion_lifting.py`: 无人类条件训练脚本
- `diffusion_copolicy/train_diffusion_lifting_human_cond.py`: 有人类条件训练脚本

### 配置文件
- `diffusion_copolicy/diffusion_policy/config/train_diffusion_transformer_lowdim_lifting.yaml`
- `diffusion_copolicy/diffusion_policy/config/train_diffusion_transformer_lowdim_lifting_human_cond.yaml`

### 核心组件
- `diffusion_copolicy/diffusion_policy/dataset/lifting_lowdim_dataset.py`: 数据集类
- `diffusion_copolicy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`: 策略类
- `diffusion_copolicy/diffusion_policy/model/diffusion/transformer_for_diffusion.py`: 模型类

## 与Flow Matching对比

| 特性 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 推理步数 | 100步 | 30步 |
| 采样器 | DDPM | ODE求解器 |
| 预测目标 | 噪声 | 速度场 |
| 训练稳定性 | 较好 | 很好 |
| 推理速度 | 较慢 | 较快 |
| 生成质量 | 很好 | 很好 |




# IRoMan: Human-Robot Collaborative Tasks

This project implements and evaluates two main collaborative tasks: **Table Carrying** and **Collaborative Lifting** using both Diffusion and Flow Matching models.

## Project Overview

This repository contains implementations for human-robot collaborative tasks with two different approaches:
- **Diffusion Co-Policy**: Based on [diffusion_copolicy](https://github.com/eleyng/diffusion_copolicy) and [table-carrying-ai](https://github.com/eleyng/table-carrying-ai)
- **Flow Matching**: An alternative approach for policy learning

## Task 1: Table Carrying

### Environment Setup
Reference implementations:
- [diffusion_copolicy](https://github.com/eleyng/diffusion_copolicy)
- [table-carrying-ai](https://github.com/eleyng/table-carrying-ai)

### Dataset and Model Locations
- **Dataset location**: `table/`
- **Trained models location**: 
  - Diffusion: `diffusion/`
  - Flow Matching: `flowmatching/`

### Evaluation Commands

#### Diffusion Model
```bash
python --run-mode hil --robot-mode planner --human-mode real --human-control joystick --render-mode gui --planner-type diffusion_policy --map-config cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml [--human-act-as-cond]
```

#### Flow Matching Model
```bash
python table-carrying-ai/scripts/test_model_flow_matching.py --run-mode hil --robot-mode planner --human-mode real --human-control joystick --render-mode gui --planner-type flowmatching --map-config cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml [--human-act-as-cond]
```

## Task 2: Collaborative Lifting

### Data Source
Training data is generated from SAC models trained on the [human-robot-gym](https://github.com/TUMcps/human-robot-gym) environment.

### Dataset Locations
- **Raw SAC data**: `collaborative-lifting-sac-350/`
- **Converted format data**: `table/`

### Flow Matching Implementation

#### Training Scripts
- **Without human condition**: `train_flow_matching_lifting.py`
- **With human condition**: `train_flow_matching_lifting_human.py`

#### Trained Models
- **Without human condition**: `flow_matching_collaborative_lifting/`
- **With human condition**: `flow_matching_collaborative_lifting_human_cond/`

#### Evaluation Scripts
- **Without human condition**: `demo_collaborative_lifting_flow_matching.py`
- **With human condition**: `demo_collaborative_lifting_flow_matching_human_condition.py`

### Diffusion Model Implementation

#### Training Scripts
- **Without human condition**: `train_diffusion_lifting.py`
- **With human condition**: `train_diffusion_lifting_human_cond.py`

#### Trained Models
- **Without human condition**: `diffusion_model_lifting_no_human_cond/`
- **With human condition**: `diffusion_model_lifting_human_cond/`

#### Evaluation Scripts
- **Without human condition**: `demo_collaborative_lifting_diffusion_simple.py`
- **With human condition**: `demo_collaborative_lifting_diffusion_human_condition.py`

## Dataset and Model Locations

### Download Pre-trained Models and Datasets
All datasets and pre-trained models are available for download:
- **Download Link**: [IRoMan.zip](https://www.dropbox.com/scl/fi/7x19y6sf7bmaw0a1sodkk/IRoMan.zip?rlkey=1j6njquxoac4s58q047e1x606&st=f3qumjdh&dl=0)

### Table Carrying Task
- **Dataset**: `diffusion_copolicy/data/table/`
- **Diffusion Models**: `table-carrying-ai/trained_models/diffusion/`
- **Flow Matching Models**: `table-carrying-ai/trained_models/flowmatching/`

### Collaborative Lifting Task
- **Raw SAC Data**: `human-robot-gym/datasets/collaborative-lifting-sac-350/`
- **Converted Data**: 
  - `diffusion_copolicy/data/table/`
  - `flow_copolicy/data/table/`
- **Flow Matching Models**: 
  - `flow_copolicy/outputs/flow_matching_collaborative_lifting/`
  - `flow_copolicy/outputs/flow_matching_collaborative_lifting_human_cond/`
- **Diffusion Models**: 
  - `diffusion_copolicy/data/outputs/diffusion_model_lifting_no_human_cond/`
  - `diffusion_copolicy/data/outputs/diffusion_model_lifting_human_cond/`

## Key Features

### Model Types
1. **Diffusion Co-Policy**: Traditional diffusion-based approach for human-robot collaboration
2. **Flow Matching**: Alternative generative modeling approach with improved efficiency

### Human Action Conditioning
Both approaches support human action conditioning:
- **Without human condition**: Models predict robot actions based on observations only
- **With human condition**: Models use human action history as additional input for better coordination

### Evaluation Modes
- **Human-in-the-loop (HIL)**: Real-time human-robot interaction
- **Simulation**: Offline evaluation with recorded human actions
- **Keyboard control**: Manual control for comparison

## Usage

### Training
1. Prepare datasets in the specified locations
2. Run appropriate training scripts for your chosen model type
3. Monitor training progress and save checkpoints

### Evaluation
1. Load trained models from the specified directories
2. Run evaluation scripts with appropriate parameters
3. Compare performance between Diffusion and Flow Matching approaches

## Dependencies

- PyTorch
- Hydra (for configuration management)
- Robosuite
- Human-robot-gym
- Diffusers (for diffusion models)
- Zarr (for data storage)

## Citation

If you use this code, please cite the original papers:
- Diffusion Co-Policy: [Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks](https://github.com/eleyng/diffusion_copolicy)
- Table Carrying: [It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying](https://github.com/eleyng/table-carrying-ai)
- Human-Robot Gym: [Human-Robot Gym](https://github.com/TUMcps/human-robot-gym)

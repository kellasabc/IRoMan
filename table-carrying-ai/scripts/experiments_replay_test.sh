#!/bin/bash
test_set=('unseen_map')
subject=('0') #$1
types=('diffusion_policy')
run_mode=('hil') #('replay_traj')
human_mode=('real') #('data')
render_mode=gui
data_dir=datasets/rnd_obstacle_v2/diffusion-training
map_dir=/home/ubuntu/IRoMan/table-carrying-ai/datasets/rnd_obstacle_v2/diffusion-training/map_cfg/ep_239.npz
map_config=cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml
human_control=joystick #[joystick, keyboard]


echo "Running diffusion_policy experiments..."

python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control ${human_control} --render-mode ${render_mode} --planner-type diffusion_policy --map-config ${map_config} --human-act-as-cond
#python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --human-act-as-cond  --render-mode ${render_mode} --planner-type diffusion_policy --data-dir ${data_dir} --map-dir ${map_dir} --map-config ${map_config}

echo "Done with diffusion_policy experiments."

#!/bin/bash
# PyCharm环境变量配置文件
# 在PyCharm中设置这些环境变量

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/.mujoco/mujoco210
export MUJOCO_PY_MJKEY_PATH=/home/ubuntu/.mujoco/mjkey.txt
export MUJOCO_GL=glfw
export PYTHONPATH=/home/ubuntu/IRoMan/human-robot-gym:$PYTHONPATH

echo "环境变量已设置："
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "MUJOCO_PY_MUJOCO_PATH: $MUJOCO_PY_MUJOCO_PATH"
echo "MUJOCO_PY_MJKEY_PATH: $MUJOCO_PY_MJKEY_PATH"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "PYTHONPATH: $PYTHONPATH"

import numpy as np
import robosuite as suite

# 用 mirror 分支的环境名
env = suite.make(
    "CollaborativeLiftingCart",      # 关键：mirror分支类名
    robots="Panda",                  # 按你的实际 demo 修改
    use_camera_obs=False,
    has_renderer=False,
    has_offscreen_renderer=False,
    control_freq=20,
    horizon=1000,
    # 如 demo 里还有其它参数（桌面尺寸/随机项等），也一并带上，确保一致
)

obs = env.reset()

# —— 起点：木板的世界坐标（刚体质心）
board_body_id = env.board_body_id
start_pos = env.sim.data.body_xpos[board_body_id].copy()
print("Board start (world):", start_pos)

# —— 目标：协同抬举没有目标位姿；API返回空列表
print("Desired goal (CollaborativeLiftingCart):", env._get_desired_goal_from_obs(obs))

# —— 推进到任务成功（或到达动画完成），记录当时坐标
success_pos = None
for _ in range(5000):
    # 随便给个零动作，只是让动画与物理向前跑；你也可以替换为手机器人保持板水平的策略动作
    action = np.zeros(env.robots[0].dof)
    obs, reward, done, info = env.step(action)
    # 这个环境的成功由动画完成判定；info里常带有类似标志
    if info.get("task_success") or info.get("animation_complete"):
        success_pos = env.sim.data.body_xpos[board_body_id].copy()
        break

print("Board position at success (world):", success_pos)

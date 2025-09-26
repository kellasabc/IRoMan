# Flow Matching 模型测试使用示例

## 自动模型选择功能

当使用以下参数组合时，程序会自动选择 `model_10Hz.ckpt` 文件进行不使用人类条件的评估：

```bash
python test_model_flow_matching.py \
    --run-mode hil \
    --robot-mode planner \
    --human-mode real \
    --human-control joystick \
    --render-mode gui \
    --planner-type flowmatching \
    --map-config cooperative_transport/gym_table/config/maps/varied_maps_test_holdout.yml
```

## 自动设置的内容

当检测到上述参数组合时，程序会自动：

1. **设置模型路径**: 自动指向 `trained_models/flowmatching/model_10Hz.ckpt`
2. **设置 human_act_as_cond = False**: 确保使用不使用人类条件的模型
3. **输出确认信息**: 显示自动设置的状态

## 输出示例

```
Human actions from: joystick. 
✓ 自动设置模型路径: /home/ubuntu/IRoMan/table-carrying-ai/trained_models/flowmatching/model_10Hz.ckpt
✓ 使用不使用人类条件的 Flow Matching 模型
✓ 设置 human_act_as_cond = False
Saving results to: /path/to/results/eval_hil_seed-42_R-planner-flow_matching_H-real-joystick
```

## 其他参数组合

对于其他参数组合，程序会使用默认的 `artifact_path` 设置，不会自动选择模型文件。

## 注意事项

- 确保 `trained_models/flowmatching/model_10Hz.ckpt` 文件存在
- 如果文件不存在，程序会显示警告信息并继续使用默认设置
- 此功能仅适用于指定的参数组合，其他组合不受影响









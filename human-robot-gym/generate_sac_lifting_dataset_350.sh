#!/bin/bash

# 生成350个episode的SAC lifting数据集
# 使用human_robot_gym的专家策略生成高质量的训练数据

echo "开始生成350个episode的SAC lifting数据集..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查配置文件是否存在
CONFIG_FILE="human_robot_gym/training/config/collaborative_lifting_sac_dataset_350.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 创建数据集目录
DATASET_DIR="../datasets/collaborative-lifting-sac-350"
if [ -d "$DATASET_DIR" ]; then
    echo "警告: 数据集目录 $DATASET_DIR 已存在"
    read -p "是否覆盖现有数据集? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有数据集..."
        rm -rf "$DATASET_DIR"
    else
        echo "取消数据集生成"
        exit 1
    fi
fi

# 生成数据集
echo "使用配置文件: $CONFIG_FILE"
echo "数据集名称: collaborative-lifting-sac-350"
echo "Episode数量: 350"
echo "线程数: 4"

python human_robot_gym/training/create_expert_dataset.py \
    --config-name collaborative_lifting_sac_dataset_350 \
    dataset_name=collaborative-lifting-sac-350 \
    n_episodes=350 \
    n_threads=1 \
    environment.horizon=3000 \
    environment.verbose=false \
    environment.has_renderer=false \
    environment.has_offscreen_renderer=false \
    environment.use_camera_obs=false \
    environment.use_object_obs=true

# 检查生成结果
if [ $? -eq 0 ]; then
    echo "数据集生成成功!"
    echo "数据集位置: $DATASET_DIR"
    
    # 显示数据集统计信息
    if [ -f "$DATASET_DIR/stats.csv" ]; then
        echo ""
        echo "数据集统计信息:"
        cat "$DATASET_DIR/stats.csv"
    fi
    
    # 显示数据集结构
    echo ""
    echo "数据集结构:"
    ls -la "$DATASET_DIR"
    
    # 显示episode数量
    EPISODE_COUNT=$(find "$DATASET_DIR" -name "ep_*" -type d | wc -l)
    echo ""
    echo "生成的episode数量: $EPISODE_COUNT"
    
    if [ "$EPISODE_COUNT" -eq 350 ]; then
        echo "✅ 成功生成350个episode"
    else
        echo "⚠️  警告: 实际生成的episode数量 ($EPISODE_COUNT) 与预期 (350) 不符"
    fi
else
    echo "❌ 数据集生成失败"
    exit 1
fi

echo ""
echo "数据集生成完成!"
echo "您可以使用以下命令查看数据集详情:"
echo "  ls -la $DATASET_DIR"
echo "  cat $DATASET_DIR/stats.csv"

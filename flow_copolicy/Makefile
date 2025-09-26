# Flow Matching Transformer 训练 Makefile

.PHONY: help test train clean install

# 默认目标
help:
	@echo "Flow Matching Transformer 训练命令:"
	@echo ""
	@echo "  make test      - 测试配置和依赖"
	@echo "  make train     - 开始训练"
	@echo "  make clean     - 清理输出文件"
	@echo "  make install   - 安装依赖"
	@echo "  make help      - 显示此帮助信息"
	@echo ""

# 测试配置
test:
	@echo "测试配置和依赖..."
	python3 test_config.py

# 开始训练
train:
	@echo "开始 Flow Matching Transformer 训练..."
	@echo "使用数据集: collaborative_lifting_sac_350.zarr"
	@echo "观察状态: 37维"
	@echo "动作状态: 10维"
	@echo "不使用 rollout 功能"
	@echo "检查点保存频率: 每10个epoch"
	python3 train_flow_matching.py

# 使用启动脚本训练
train-script:
	@echo "使用启动脚本开始训练..."
	./run_training.sh

# 清理输出文件
clean:
	@echo "清理输出文件..."
	rm -rf outputs/flow_matching_collaborative_lifting/checkpoints/*
	rm -rf outputs/flow_matching_collaborative_lifting/logs.json.txt
	@echo "清理完成!"

# 安装依赖
install:
	@echo "安装依赖包..."
	pip install torch hydra-core omegaconf wandb tqdm numpy
	@echo "依赖安装完成!"

# 检查数据集
check-data:
	@echo "检查数据集..."
	@if [ -d "data/table/collaborative_lifting_sac_350.zarr" ]; then \
		echo "✓ 数据集存在: data/table/collaborative_lifting_sac_350.zarr"; \
		echo "  观察状态: 37维"; \
		echo "  动作状态: 10维"; \
		echo "  过去动作: 6维"; \
		echo "  Episode数量: 350"; \
		ls -la data/table/collaborative_lifting_sac_350.zarr/; \
	else \
		echo "✗ 数据集不存在: data/table/collaborative_lifting_sac_350.zarr"; \
		exit 1; \
	fi

# 显示训练状态
status:
	@echo "训练状态检查..."
	@if [ -d "outputs/flow_matching_collaborative_lifting" ]; then \
		echo "✓ 输出目录存在"; \
		echo "  检查点数量: $$(ls -1 outputs/flow_matching_collaborative_lifting/checkpoints/*.pt 2>/dev/null | wc -l)"; \
		echo "  最新日志: $$(ls -la outputs/flow_matching_collaborative_lifting/logs.json.txt 2>/dev/null || echo '无日志文件')"; \
	else \
		echo "✗ 输出目录不存在"; \
	fi

# 快速测试（小规模训练）
quick-test:
	@echo "快速测试模式..."
	@echo "修改配置为调试模式..."
	@sed -i 's/n_epochs: 500/n_epochs: 2/' flow_policy/config/train_flow_matching_collaborative_lifting.yaml
	@sed -i 's/checkpoint_every: 10/checkpoint_every: 1/' flow_policy/config/train_flow_matching_collaborative_lifting.yaml
	@echo "开始快速测试..."
	python3 train_flow_matching.py
	@echo "恢复原始配置..."
	@sed -i 's/n_epochs: 2/n_epochs: 500/' flow_policy/config/train_flow_matching_collaborative_lifting.yaml
	@sed -i 's/checkpoint_every: 1/checkpoint_every: 10/' flow_policy/config/train_flow_matching_collaborative_lifting.yaml
	@echo "快速测试完成!"

# 显示当前配置
show-config:
	@echo "当前训练配置:"
	@echo "  训练轮数: $$(grep 'n_epochs:' flow_policy/config/train_flow_matching_collaborative_lifting.yaml | head -1 | awk '{print $$2}')"
	@echo "  检查点频率: $$(grep 'checkpoint_every:' flow_policy/config/train_flow_matching_collaborative_lifting.yaml | head -1 | awk '{print $$2}')"
	@echo "  验证频率: $$(grep 'eval_every:' flow_policy/config/train_flow_matching_collaborative_lifting.yaml | head -1 | awk '{print $$2}')"
	@echo "  采样频率: $$(grep 'sample_every:' flow_policy/config/train_flow_matching_collaborative_lifting.yaml | head -1 | awk '{print $$2}')"
	@echo "  观察维度: $$(grep 'obs_dim:' flow_policy/config/task/collaborative_lifting_lowdim.yaml | head -1 | awk '{print $$2}')"
	@echo "  动作维度: $$(grep 'action_dim:' flow_policy/config/task/collaborative_lifting_lowdim.yaml | head -1 | awk '{print $$2}')"

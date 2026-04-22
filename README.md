# 基于 PER-DQN 的多环境策略决策智能体研究与实现

本项目使用 Python + PyTorch + Gymnasium 实现 DQN 与 PER-DQN 两种强化学习算法，并在三个环境中开展对比实验：Taxi-v3、MountainCar-v0、简化 Dinosaur Game。

项目目标是为本科毕业设计提供一个结构清晰、可运行、便于实验复现和论文写作的强化学习代码基础。

## 项目特点

- 支持 DQN 与 PER-DQN 两种算法
- 支持 Taxi-v3、MountainCar-v0、自定义 Dino 环境
- 统一使用低维状态向量与 MLP Q 网络
- 训练、评估、批量实验、自动绘图入口相互分离
- 自动保存日志、模型参数与对比图像
- 对关键模块补充了适量中文注释，便于课程设计与论文答辩说明

## 项目结构

```text
PERDQN_Strategy_Decision/
├── train.py               # 单环境训练入口
├── eval.py                # 单环境评估入口
├── experiment.py          # 批量实验入口
├── compare_plots.py       # 自动对比绘图入口
├── main.py                # 兼容提示入口
├── config.py              # 参数与路径配置
├── DQN/
│   ├── __init__.py
│   ├── training.py
│   ├── evaluation.py
│   ├── experiment_utils.py
│   ├── agents/
│   ├── buffers/
│   ├── envs/
│   ├── models/
│   └── utils/
├── results/
│   ├── models/
│   ├── logs/
│   └── figures/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境说明

### 1. Taxi-v3
- 离散状态空间
- 离散动作空间
- 状态通过 one-hot 编码输入网络
- 适合体现 PER-DQN 对关键经验的优先学习能力

### 2. MountainCar-v0
- 连续二维状态空间
- 离散动作空间
- 状态经过归一化后输入网络
- 适合分析 PER-DQN 在稀疏奖励和长期策略任务中的作用与局限

### 3. Dinosaur Game
- 自定义 Gymnasium 风格环境
- 使用低维状态向量，不使用图像输入
- 动作包括：0 不操作，1 跳跃
- 状态包括：y 坐标、竖直速度、是否落地、障碍物距离、障碍物高度、障碍物宽度、游戏速度
- 环境中使用简化动力学、奖励设计与碰撞检测，便于解释关键经验的来源

## 算法说明

### DQN
- Q Network + Target Network
- Replay Buffer
- epsilon-greedy
- Huber Loss
- 周期性更新目标网络

### PER-DQN
- 在 DQN 基础上引入 Prioritized Replay Buffer
- 根据 TD-error 赋予样本优先级
- 采样时引入 Importance Sampling 权重
- 使用 alpha 和 beta 控制采样分布与偏差修正

## 三环境下 PER-DQN 的预期表现

### Taxi-v3
- 关键经验包括成功 pickup、成功 dropoff、错误 pickup/dropoff
- PER-DQN 预期更快提升平均奖励并减少完成步数
- 在较简单任务中，最终性能提升可能有限

### MountainCar-v0
- 关键经验包括接近目标位置、有效积累速度、成功到达山顶
- PER-DQN 可提高关键转移样本利用率
- 但如果探索不足，PER-DQN 也无法独立解决学习困难问题

### Dinosaur Game
- 关键经验包括接近障碍物、成功越过障碍物、碰撞失败
- PER-DQN 预期更快提高平均存活时间和通过障碍物数量
- 自定义环境结论会受到奖励设计和状态设计影响

## 安装方式

```bash
pip install -r requirements.txt
```

## 配置方式

本项目默认不使用命令行参数，直接在 [config.py](config.py) 中修改：

### 训练入口配置
- `TRAIN_CONFIG["env_name"]`：`taxi` / `mountaincar` / `dino`
- `TRAIN_CONFIG["algo_name"]`：`dqn` / `perdqn`
- `TRAIN_CONFIG["render"]`：训练时是否渲染
- `TRAIN_CONFIG["plot_after_train"]`：训练结束后是否尝试自动绘图

### 评估入口配置
- `EVAL_CONFIG["env_name"]`
- `EVAL_CONFIG["algo_name"]`
- `EVAL_CONFIG["render"]`
- `EVAL_CONFIG["use_best_model"]`

### 批量实验配置
- `EXPERIMENT_CONFIG["env_names"]`
- `EXPERIMENT_CONFIG["algo_names"]`
- `EXPERIMENT_CONFIG["seeds"]`

### 自动绘图配置
- `PLOT_CONFIG["env_names"]`
- `PLOT_CONFIG["window"]`

## 运行方式

### 1. 单环境训练
```bash
python train.py
```

### 2. 单环境评估
```bash
python eval.py
```

### 3. 批量实验
```bash
python experiment.py
```

### 4. 自动生成对比图
```bash
python compare_plots.py
```

## 推荐实验流程

建议按下面顺序完成论文实验：

1. 在 `config.py` 中设置单环境训练参数
2. 使用 `python train.py` 分别训练 DQN 与 PER-DQN
3. 使用 `python eval.py` 评估模型效果
4. 使用 `python experiment.py` 运行多随机种子批量实验
5. 使用 `python compare_plots.py` 统一生成对比图
6. 根据 `results/logs/`、`results/models/`、`results/figures/` 整理论文结果

## 默认参数说明

当前默认参数针对论文实验做了适度调整：

- `moving_average_window = 50`：更适合绘制论文中的平滑训练曲线
- `test_episodes = 20`：评估结果更稳定
- Taxi 默认 `episodes = 1500`
- MountainCar 默认 `episodes = 2000`
- Dino 默认 `episodes = 1800`
- 按环境分别设置 `min_replay_size` 和 `epsilon_decay`，更符合不同任务难度

## 输出结果

### 日志
保存在：
- `results/logs/{env_name}_{algo_name}_train_log.csv`
- 批量实验会额外生成：`results/logs/experiment_summary.csv`

单次训练日志字段包括：
- `episode`
- `total_reward`
- `steps`
- `epsilon`
- `loss`
- `success`
- `custom_metric`

### 模型
保存在：
- `results/models/{env_name}_{algo_name}_best.pth`
- `results/models/{env_name}_{algo_name}_final.pth`
- 批量实验模型会追加 `seed` 后缀

### 图像
保存在：
- `results/figures/`

包括：
- 奖励对比曲线
- 步数对比曲线
- loss 对比曲线
- success rate 或障碍物通过数量对比曲线

## 代码设计说明

### 为什么把实现统一放入 `DQN/` 包
这样可以把入口文件和核心实现分开，便于：
- 组织训练、评估、批量实验等不同流程
- 统一管理导入路径
- 提高项目可读性

### 为什么继续使用低维状态 + MLP
这是为了保证：
- 实现简单稳定
- 三个环境能共用一套网络结构
- 更适合作为本科毕业设计项目而不是复杂工业级框架

### 为什么 PER 使用简单概率数组实现
本项目更强调实验可解释性与代码简洁性，因此没有引入更复杂的 SumTree 实现。对于当前实验规模，这种实现已经足够支持论文中的对比实验。

## 论文实验分析建议

建议从以下角度分析 DQN 与 PER-DQN：
- 平均奖励变化趋势
- 收敛速度
- 平均步数变化
- 成功率或通过障碍物数量
- PER-DQN 在关键经验利用率上的优势
- PER-DQN 在探索不足和稀疏奖励环境中的局限

## 说明

本项目强调代码清晰、稳定、可运行，适合本科毕业设计，不追求复杂工业级框架设计。若后续需要，还可以在此基础上继续补充更完整的批量统计、平均曲线汇总和实验表格输出。
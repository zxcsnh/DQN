# DQN / PER-DQN Strategy Decision Experiments

本项目用于比较 DQN 与 Prioritized Experience Replay DQN（PER-DQN）在离散动作决策任务中的表现。当前代码支持三个环境：

- `dino`：自定义 Chrome Dino / T-Rex Gymnasium 环境
- `taxi`：Gymnasium Taxi 环境
- `mountaincar`：Gymnasium MountainCar 环境

说明：本文档以当前代码为准。若历史实验结果或旧文档与源码不一致，请优先参考 `config.py`、`DQN/training.py` 和具体环境实现。

## 项目结构

```text
.
|-- config.py                         # 全局配置、环境超参数、结果目录
|-- train.py                          # 单次训练入口
|-- eval.py                           # 单个模型评估入口
|-- experiment.py                     # 批量实验入口
|-- summary.py                        # 从日志生成实验汇总
|-- compare_plots.py                  # DQN / PER-DQN 对比图生成
|-- plot_log.py                       # 单次训练日志绘图
|-- DQN/
|   |-- training.py                   # 训练主循环
|   |-- evaluation.py                 # 评估与最终测试
|   |-- shared.py                     # 名称校验、Agent 工厂、任务指标
|   |-- q_network.py                  # Q 网络
|   |-- experiment_utils.py           # 批量实验与汇总工具
|   |-- agents/
|   |   |-- dqn_agent.py              # DQN Agent
|   |   `-- perdqn_agent.py           # PER-DQN Agent
|   |-- buffers/
|   |   |-- replay_buffer.py          # 均匀经验回放
|   |   `-- prioritized_replay_buffer.py
|   |-- envs/
|   |   |-- env_factory.py            # 环境工厂
|   |   `-- dino/
|   |       |-- env.py                # 自定义 TrexEnv-v0
|   |       |-- main.py               # 人类游玩入口
|   |       `-- sprites/              # Dino 资源文件
|   `-- utils/
|       |-- state_processor.py        # 状态预处理
|       |-- seed_utils.py             # 随机种子
|       |-- logger.py                 # CSV 日志
|       `-- plot_utils.py             # 绘图工具
`-- results/                          # 训练输出、模型、日志、图表
```

## 安装依赖

项目使用 Python 3.12+，依赖由 `pyproject.toml` 管理。

```powershell
uv sync
```

主要依赖包括：

- `torch`
- `gymnasium[other]`
- `numpy`
- `pygame`
- `matplotlib`
- `opencv-python`
- `tqdm`

## 快速开始

单次训练：

```powershell
python train.py
```

当前 `train.py` 默认运行：

```python
train(
    env_name="dino",
    algo_name="dqn",
    render=False,
    plot_after_train=True,
)
```

评估指定模型：

```powershell
python eval.py
```

批量实验：

```powershell
python experiment.py
```

生成实验汇总：

```powershell
python summary.py results\0513-2330-experiment
```

生成对比图：

```powershell
python compare_plots.py
```

## 算法实现

### DQN

`DQNAgent` 位于 `DQN/agents/dqn_agent.py`，包含：

- epsilon-greedy 探索
- 经验回放
- target Q-network
- Huber loss（`SmoothL1Loss`）
- 梯度裁剪
- 可选 Double DQN
- 可选 soft target update
- 可选 warmup 随机探索阶段

Q 网络定义在 `DQN/q_network.py`，当前结构为：

```text
Linear -> LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU -> Linear
```

### PER-DQN

`PERDQNAgent` 继承 `DQNAgent`，将普通 replay buffer 替换为 `PrioritizedReplayBuffer`。

当前 PER 实现包含：

- SumTree 优先级采样
- TD error 更新优先级
- importance sampling weights
- beta 从 `beta_start` 线性退火到 `1.0`

优先级公式：

```text
priority = (abs(td_error) + priority_epsilon) ** alpha
```

IS 权重会除以 batch 内最大权重，使最大权重归一化为 `1.0`。

## 当前环境配置

配置集中在 `config.py`。

### 通用配置字段

`Config` 中的主要字段包括：

- `episodes`
- `max_steps_per_episode`
- `final_test_episodes`
- `eval_interval_episodes`
- `eval_episodes`
- `batch_size`
- `seed`
- `gamma`
- `learning_rate`
- `hidden_dim`
- `epsilon_start`
- `epsilon_end`
- `epsilon_decay_steps`
- `replay_buffer_size`
- `min_replay_size`
- `target_update_freq`
- `soft_target_update`
- `target_update_tau`
- `success_threshold`
- `gradient_clip_norm`
- `use_double_dqn`
- `warmup_steps`

### 环境默认值

| 环境 | env_id | episodes | max steps | batch | lr | hidden | buffer | min replay | target update | Double DQN | soft target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `taxi` | `Taxi-v4` | 2000 | 200 | 64 | `5e-4` | 128 | 30000 | 500 | 500 | False | False |
| `mountaincar` | `MountainCar-v0` | 2000 | 500 | 128 | `5e-4` | 128 | 100000 | 5000 | 1000 | False | False |
| `dino` | `TrexEnv-v0` | 3000 | 8000 | 128 | `1e-5` | 256 | 200000 | 5000 | 2000 | False | True |

注意：当前代码里 `taxi` 配置为 `Taxi-v4`。如果本地 Gymnasium 只提供 `Taxi-v3`，需要在 `config.py` 中改为 `Taxi-v3` 后再运行 Taxi 实验。

### PER 默认值

`PerConfig` 当前默认值：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `alpha` | 0.5 | 优先级强度 |
| `beta_start` | 0.4 | IS 权重 beta 初始值 |
| `beta_anneal_steps` | 180000 | beta 退火步数 |
| `priority_epsilon` | `1e-4` | 防止零优先级 |

## Dino / T-Rex 环境

自定义环境位于 `DQN/envs/dino/env.py`，注册为 `TrexEnv-v0`。

### 动作空间

动作空间为 `Discrete(3)`：

| Action | 含义 |
| ---: | --- |
| 0 | noop |
| 1 | jump |
| 2 | duck |

### 观测空间

当前 Dino 观测是 23 维连续向量，定义在 `TrexEnv._build_observation()` 中。状态会在 `DQN/utils/state_processor.py` 中按 `observation_space.low/high` 做 min-max 归一化。

基础特征：

| 索引 | 特征 |
| ---: | --- |
| 0 | dino 距离地面的高度 |
| 1 | 垂直速度 |
| 2 | 是否正在跳跃 |
| 3 | 是否正在下蹲 |
| 4 | 当前游戏速度 |
| 5 | 当前是否可以跳跃 |
| 6 | 最近两个障碍物间距 |

随后分别追加最近两个障碍物的 8 个特征：

| 特征 | 含义 |
| --- | --- |
| `obstacle_present` | 是否存在该障碍 |
| `obstacle_type` | `0=cactus`，`1=ptera` |
| `distance` | 障碍物到 dino 右侧的水平距离 |
| `time_to_collision` | 按当前速度估算的碰撞时间 |
| `width` | 障碍物宽度 |
| `height` | 障碍物高度 |
| `center_y` | 障碍物中心 y 坐标 |
| `relative_y` | 障碍物中心与 dino 中心的相对 y 坐标 |

因此总维度为：

```text
7 + 8 * 2 = 23
```

### 奖励设计

当前奖励逻辑在 `TrexEnv._compute_reward()` 中：

| 事件 | 奖励 |
| --- | ---: |
| 每步存活 | `+0.01` |
| 通过障碍物 | `0.0 * newly_cleared` |
| 碰撞死亡 | `-1.0` |
| 空中重复跳跃 | `0.0` |

`obstacles_cleared` 仍会被统计，并作为 Dino 的自定义任务指标。成功判定为：

```text
obstacles_cleared >= config.success_threshold
```

当前 Dino 的 `success_threshold` 为 `20`。

### 难度递增

Dino 每 700 帧提升一次速度：

```text
gamespeed += 1
```

地面和障碍物移动速度会同步加快。

## 状态预处理

`DQN/utils/state_processor.py` 对不同环境做统一处理：

- Taxi：离散状态转 one-hot
- MountainCar：按固定上下界归一化到 `[0, 1]`
- Dino：按环境 `observation_space.low/high` 归一化到 `[0, 1]`

## 日志与输出

每次训练会创建独立 run 目录：

```text
results/<timestamp>-<env>-<algo>-seed<seed>/
|-- logs/
|-- models/
`-- figures/
```

训练日志字段包括：

- `episode`
- `total_reward`
- `steps`
- `epsilon`
- `loss`
- `success`
- `custom_metric`
- `reward_per_step`
- `eval_avg_reward`
- `eval_success_rate`
- PER 相关指标：`per_beta`、`per_mean_abs_td_error`、`per_mean_weight` 等
- Dino 相关指标：`score`、`speed`、`obstacles_cleared`
- MountainCar 相关指标：`max_position`

训练会保存：

- best model：按定期评估平均奖励选择
- final model：训练结束时的模型

如果 `run_final_test=True`，训练结束后会分别测试 best 和 final 模型。

## 实验结果解读

建议重点比较：

- 平均奖励曲线
- 平均步数曲线
- success rate
- Dino 的 `obstacles_cleared`
- MountainCar 的 `max_position`
- PER 的 TD error 与 IS weight 指标

当前 Dino 奖励中没有通过障碍物的额外正奖励，因此 `total_reward` 主要反映存活时长；如果论文目标强调清障能力，建议同时报告 `obstacles_cleared`，不要只看 reward。

## 已知注意事项

- `README.md` 已按当前源码更新；旧实验说明中的 15 维 Dino 观测、过障碍 `+2`、死亡 `-10` 等不再符合当前代码。
- `taxi` 当前配置为 `Taxi-v4`，部分 Gymnasium 版本只提供 `Taxi-v3`。
- `compare_plots.py`、`eval.py`、`plot_log.py` 中存在一些面向本机历史结果的默认路径，复现实验时可能需要改成自己的 run 目录。
- Windows 路径字符串建议使用 raw string 或正斜杠，避免 `SyntaxWarning: invalid escape sequence`。

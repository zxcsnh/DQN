# PER-DQN 策略决策实验

本项目用于本科毕业设计中的「基于 PER-DQN 的策略决策」实验。实现了 DQN 与 Prioritized Experience Replay DQN（PER-DQN），支持 Double DQN 可选扩展，并在 Taxi-v3、MountainCar-v0 与自定义 Dino/T-Rex 三类离散动作决策环境中进行对比。

## 项目结构

```
├── config.py                    # 全局配置（超参数、环境配置、路径常量）
├── train.py                     # 单次训练入口
├── eval.py                      # 模型评估入口
├── experiment.py                # 批量实验入口（多环境 × 多算法 × 多种子）
├── compare_plots.py             # DQN vs PER-DQN 对比曲线生成
│
├── DQN/
│   ├── q_network.py             # Q 网络（3 层 MLP）
│   ├── training.py              # 训练主循环
│   ├── evaluation.py            # 模型评估逻辑
│   ├── experiment_utils.py      # 批量实验与结果汇总
│   ├── shared.py                # 名称校验、Agent 工厂、指标计算
│   │
│   ├── agents/
│   │   ├── dqn_agent.py         # DQN Agent（支持 Double DQN 与 warmup）
│   │   └── perdqn_agent.py      # PER-DQN Agent（继承 DQN，覆盖优先回放逻辑）
│   │
│   ├── buffers/
│   │   ├── replay_buffer.py     # 均匀采样经验回放池（deque）
│   │   └── prioritized_replay_buffer.py  # SumTree 优先经验回放池 + IS 权重
│   │
│   ├── envs/
│   │   ├── env_factory.py       # 环境工厂（Taxi / MountainCar / Dino）
│   │   ├── dino_env.py          # 简化版 Dino 环境（无 Pygame，用于快速验证）
│   │   └── dino/
│   │       ├── env.py           # 完整 TrexEnv（Pygame 渲染、精灵动画、音效）
│   │       └── main.py          # 人类可玩入口
│   │
│   └── utils/
│       ├── logger.py            # CSV 日志记录器
│       ├── plot_utils.py        # 单次训练曲线 + DQN vs PER-DQN 对比图
│       ├── seed_utils.py        # 全局随机种子与环境种子
│       └── state_processor.py   # 状态预处理（One-hot、归一化）
│
└── results/
    ├── models/                  # 模型检查点（best.pth / final.pth）
    ├── logs/                    # 训练日志 CSV
    └── figures/                 # 对比图输出
```

## 算法设计

### DQN（Deep Q-Network）

使用 MLP 近似动作价值函数 Q(s, a)，通过目标网络（target network）缓解训练不稳定性。核心要素：

- **Q 网络**：3 层全连接网络（`Linear → ReLU → Linear → ReLU → Linear`）
- **目标网络**：与 Q 网络结构相同，定期硬同步（`target_update_freq`）
- **探索策略**：ε-greedy，线性衰减（`epsilon_start → epsilon_end`）
- **损失函数**：SmoothL1Loss（Huber loss）
- **梯度裁剪**：`clip_grad_norm_` 防止梯度爆炸
- **随机探索热身**：训练前 `warmup_steps` 步使用纯随机策略填充经验池

### Double DQN（可配置选项）

在 `config.py` 中通过 `use_double_dqn` 开关控制。启用后，目标 Q 值的计算由：

```
target = r + γ · max_a' Q_target(s', a')
```

改为：

```
a* = argmax_a' Q_online(s', a')
target = r + γ · Q_target(s', a*)
```

Online 网络负责选择动作，目标网络负责评估，有效缓解 Q 值过估计问题。Dino 环境默认启用。

### PER-DQN（Prioritized Experience Replay DQN）

在 DQN 基础上引入优先经验回放，核心改进：

1. **SumTree 优先级采样**（O(log n)）：根据 TD error 为每条经验分配优先级，使高误差、高信息量的经验被更频繁重放。
2. **优先级计算**（标准 PER 公式）：`priority = (|TD_error| + ε_priority)^α`
   - α 控制优先级程度（α=0 退化为均匀采样，α=1 为完全按 TD error 优先级）
3. **重要性采样权重**（IS weights）：`w_i = (N · P(i))^(-β) / max_j(w_j)`
   - β 从 `beta_start` 线性递增至 1.0，修正非均匀采样引入的分布偏差
   - 采用 `1 / max(w)` 归一化，确保权重仅向下缩放更新（标准 PER 方法）
4. **新经验优先**：新存入的经验被赋予当前最大优先级，保证每条经验至少被采样一次

## Dino 策略决策环境

自定义 Dino 环境已注册为 Gymnasium 环境 `TrexEnv-v0`，通过 `gym.make("TrexEnv-v0")` 创建，与标准 Gymnasium API 完全兼容。

### 动作空间

| Action | 名称 | 策略角色 |
| --- | --- | --- |
| `0` | noop | 保持当前状态，用于等待时机或保持不动 |
| `1` | jump | 跳跃越过地面障碍（cactus） |
| `2` | duck | 下蹲躲避空中障碍（ptera） |

### 观测空间

15 维连续向量，包含角色运动状态与最近两个障碍物信息：

| 索引 | 特征 | 含义 |
| --- | --- | --- |
| `0` | dino height | 恐龙底部距地面高度 |
| `1` | vertical velocity | 垂直速度（正值下落，负值上升） |
| `2` | is jumping | 是否处于跳跃状态 |
| `3` | is ducking | 是否处于下蹲状态 |
| `4` | game speed | 当前游戏速度（初始 4，每 700 帧 +1） |
| `5` | obstacle 1 type | 最近障碍物类型（0=地面 cactus，1=空中 ptera） |
| `6` | obstacle 1 distance | 最近障碍物与恐龙右侧的水平距离 |
| `7` | obstacle 1 width | 最近障碍物宽度 |
| `8` | obstacle 1 height | 最近障碍物高度 |
| `9` | obstacle 1 center y | 最近障碍物中心纵坐标 |
| `10` | obstacle 2 type | 第二近障碍物类型 |
| `11` | obstacle 2 distance | 第二近障碍物距离 |
| `12` | obstacle 2 width | 第二近障碍物宽度 |
| `13` | obstacle 2 height | 第二近障碍物高度 |
| `14` | obstacle 2 center y | 第二近障碍物中心纵坐标 |

所有观测值经 min-max 归一化至 [0, 1] 后输入网络。

### 奖励设计

| 事件 | 奖励 | 设计动机 |
| --- | --- | --- |
| 每步存活 | `+0.1` | 提供稠密生存信号，鼓励持续存活 |
| 清除一个障碍物 | `+2.0` | 任务核心收益，显著高于单步存活奖励 |
| 空中重复跳跃 | `-0.02` | 微小惩罚，抑制无效跳跃（spam jump） |
| 碰撞死亡 | `-10.0` | 惩罚失败策略，强化避障学习 |

奖励层级：清除障碍物 (+2.0) ≫ 存活 (+0.1/步) > 避免无效跳跃 (-0.02)

成功指标：单回合清除障碍物数量 ≥ `success_threshold`（Dino 环境默认 15）记为成功。

### 难度递增

游戏速度每 700 帧增加 1（初始 4），障碍物与地面移动速度随之增加，要求智能体逐步适应越来越快的决策节奏。

## 超参数配置

所有超参数集中在 `config.py` 中，通过 frozen dataclass `Config` 管理。每个环境有独立的配置实例。

### 通用超参数（Config 默认值）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `episodes` | 1500 | 训练回合数 |
| `max_steps_per_episode` | 200 | 单回合最大步数（截断条件） |
| `batch_size` | 64 | 每次更新的批次大小 |
| `gamma` | 0.99 | 折扣因子 |
| `learning_rate` | 5e-4 | Adam 优化器学习率 |
| `hidden_dim` | 128 | Q 网络隐藏层维度 |
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_end` | 0.05 | 最终探索率 |
| `epsilon_decay_steps` | 10000 | ε 线性衰减步数（按 train_steps 计算） |
| `replay_buffer_size` | 50000 | 经验回放池容量 |
| `min_replay_size` | 500 | 开始训练前缓冲区最小经验数 |
| `target_update_freq` | 1000 | 目标网络同步频率（按 train_steps） |
| `gradient_clip_norm` | 10.0 | 梯度裁剪阈值 |
| `use_double_dqn` | False | 是否启用 Double DQN |
| `warmup_steps` | 0 | 随机探索热身步数（按 env_steps 计算） |
| `eval_interval_episodes` | 50 | 评估间隔（回合数） |
| `eval_episodes` | 10 | 每次评估的回合数 |

### PER 专属超参数（PerConfig）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `alpha` | 0.6 | 优先级指数（0=均匀，1=完全 TD error 优先） |
| `beta_start` | 0.4 | IS 权重 β 初始值 |
| `beta_increment` | 5e-5 | 每次更新后 β 的增量（最终至 1.0） |
| `priority_epsilon` | 1e-5 | 优先级平滑小量（防止零优先级） |

### 环境专属配置

| 参数 | Taxi-v3 | MountainCar-v0 | TrexEnv-v0 |
| --- | --- | --- | --- |
| `episodes` | 1800 | 3000 | 2000 |
| `max_steps_per_episode` | 200 | 200 | 8000 |
| `batch_size` | 64 | 128 | 128 |
| `gamma` | 0.99 | 0.99 | 0.995 |
| `learning_rate` | 3e-4 | 1e-3 | 1e-4 |
| `hidden_dim` | 128 | 128 | 256 |
| `epsilon_decay_steps` | 8000 | 20000 | 30000 |
| `replay_buffer_size` | 30000 | 100000 | 100000 |
| `min_replay_size` | 500 | 2000 | 5000 |
| `target_update_freq` | 500 | 1000 | 2000 |
| `gradient_clip_norm` | 10.0 | 10.0 | 5.0 |
| `success_threshold` | 1 | 1 | 15 |
| `use_double_dqn` | False | False | **True** |
| `warmup_steps` | 0 | 0 | **5000** |

## 运行实验

### 安装依赖

```powershell
uv sync
```

### 单次训练

```powershell
python train.py
```

默认环境与算法可在 `train.py` 中修改（当前默认 `dino` + `perdqn`）。

### 模型评估

```powershell
python eval.py
```

加载训练保存的最佳模型，使用 ε=0 进行确定性评估。

### 批量对比实验

```powershell
python experiment.py
```

遍历 `[taxi, mountaincar, dino] × [dqn, perdqn] × [42, 52, 62]` 共 18 组实验，结果汇总至 `results/logs/experiment_summary.csv`。

### 生成对比图

```powershell
python compare_plots.py
```

基于训练日志生成 DQN 与 PER-DQN 的奖励、步数、损失和任务指标对比曲线，保存至 `results/figures/`。

## 论文预期证据

建议在论文中展示以下实验结果：

- 三个环境下 DQN 与 PER-DQN 的**平均奖励曲线**（含多种子均值与标准差）
- **成功率或任务指标曲线**：Taxi/MountainCar 使用 success rate，Dino 使用 `obstacles_cleared`
- **收敛速度对比**：PER-DQN 是否在更少的环境交互步数下达到同等或更高性能
- **消融分析**（可选）：
  - Double DQN 对 Dino 环境的过估计缓解效果
  - warmup 阶段对训练初期稳定性的影响
  - α 与 β 超参数对 PER 性能的敏感性

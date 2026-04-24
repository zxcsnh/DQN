# PER-DQN Strategy Decision Experiments

本项目用于本科毕业设计中的“基于 PER-DQN 的策略决策”实验。代码实现了 DQN 与 Prioritized Experience Replay DQN（PER-DQN），并在 Taxi、MountainCar 与 Dino/T-Rex 三类离散动作决策环境中进行对比。

## Project Structure

- `DQN/agents/`：DQN 与 PER-DQN 智能体实现。
- `DQN/buffers/`：普通经验回放池与优先经验回放池。
- `DQN/envs/`：Gymnasium 环境工厂与自定义 Dino 环境。
- `DQN/utils/`：状态预处理、随机种子、CSV 日志与绘图工具。
- `train.py`：单次训练入口。
- `eval.py`：模型评估入口。
- `experiment.py`：多环境、多算法、多随机种子的批量实验入口。
- `compare_plots.py`：根据训练日志生成 DQN 与 PER-DQN 对比曲线。

## Algorithm Design

DQN 使用 Q 网络近似动作价值函数，并通过目标网络缓解训练不稳定问题。PER-DQN 在 DQN 基础上引入优先经验回放：根据 TD error 为样本分配优先级，使智能体更频繁学习高误差、高信息量的经验；同时使用重要性采样权重修正优先采样带来的分布偏差。

PER-DQN 的核心实现位于：

- `DQN/agents/perdqn_agent.py`：TD error、重要性采样权重、优先级更新与 beta 递增。
- `DQN/buffers/prioritized_replay_buffer.py`：优先采样概率、样本权重与优先级维护。

## Dino Strategy Decision Environment

自定义 Dino 环境已注册为 Gymnasium 环境 `TrexEnv-v0`。训练框架通过 `gym.make("TrexEnv-v0")` 创建环境，保持与标准 Gymnasium 环境一致的 `reset` 和 `step` 接口。

### Action Space

| Action | Meaning | Strategy role |
| --- | --- | --- |
| `0` | `noop` | 保持当前状态，用于等待、落地或避免不必要跳跃。 |
| `1` | `jump` | 跳跃越过地面障碍，如 cactus。 |
| `2` | `duck` | 下蹲躲避空中障碍，如 ptera。 |

### Observation Space

Dino 状态为 15 维连续向量，表示角色运动状态与最近两个障碍物信息。

| Index | Feature | Meaning |
| --- | --- | --- |
| `0` | dino height | 恐龙相对地面的高度。 |
| `1` | vertical velocity | 恐龙垂直速度，用于判断上升、下落或落地。 |
| `2` | is jumping | 是否处于跳跃状态。 |
| `3` | is ducking | 是否处于下蹲状态。 |
| `4` | game speed | 当前游戏速度，速度越高决策时间越短。 |
| `5` | obstacle 1 type | 最近障碍类型，`0` 表示地面障碍，`1` 表示空中障碍。 |
| `6` | obstacle 1 distance | 最近障碍物与恐龙的水平距离。 |
| `7` | obstacle 1 width | 最近障碍物宽度。 |
| `8` | obstacle 1 height | 最近障碍物高度。 |
| `9` | obstacle 1 center y | 最近障碍物中心纵坐标。 |
| `10` | obstacle 2 type | 第二近障碍类型。 |
| `11` | obstacle 2 distance | 第二近障碍物与恐龙的水平距离。 |
| `12` | obstacle 2 width | 第二近障碍物宽度。 |
| `13` | obstacle 2 height | 第二近障碍物高度。 |
| `14` | obstacle 2 center y | 第二近障碍物中心纵坐标。 |

### Reward Design

| Event | Reward | Purpose |
| --- | --- | --- |
| Survive one step | `+0.1` | 鼓励智能体持续存活。 |
| Clear one obstacle | `+1.0` | 鼓励成功通过障碍物，是主要任务收益。 |
| Collision | `-10.0` | 惩罚失败策略，强化避障决策。 |
| Reach max steps | Episode truncated | 代表该回合达到最大步数上限。 |

成功指标使用 `obstacles_cleared`。当单回合清除障碍数量达到 `config.py` 中的 `success_threshold` 时，该回合记为成功。

## Running Experiments

Install dependencies:

```powershell
uv sync
```

Run a single training job:

```powershell
python train.py
```

Run the full DQN vs PER-DQN experiment:

```powershell
python experiment.py
```

Generate comparison plots after both DQN and PER-DQN logs exist:

```powershell
python compare_plots.py
```

## Expected Thesis Evidence

论文中建议至少展示以下结果：

- 三个环境下 DQN 与 PER-DQN 的奖励曲线。
- 成功率或任务指标曲线：Taxi/MountainCar 使用 success，Dino 使用 `obstacles_cleared`。
- 多随机种子实验的均值与标准差。
- PER-DQN 相比 DQN 的收敛速度、稳定性与最终性能分析。

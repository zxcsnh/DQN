# 面向低维状态空间的 DQN / PER-DQN 实验项目

这是一个基于 PyTorch 的强化学习项目，用于在低成本环境上对比 `DQN` 与 `PER-DQN`，并为本科毕业设计中的策略决策分析提供实验基础。

当前项目不再面向 Atari 图像输入，也不再使用图像预处理、卷积网络或帧堆叠流程。现在的默认范式是：

- 低维状态输入
- MLP Q 网络
- 经验回放 / 优先经验回放
- 单次训练、批量实验、结果汇总、模型回放

当前优先支持的环境包括：

- `CartPole-v1`：基础算法验证
- `Taxi-v3`：主要策略决策实验
- `MountainCar-v0`：补充稀疏奖励实验

后续也可以扩展到自定义小游戏环境，例如谷歌小恐龙，但建议继续采用**特征向量状态**而不是像素图像输入。

## 当前功能

- `DQN`
  - 目标网络
  - experience replay
  - epsilon-greedy 探索
- `PER-DQN`
  - 优先经验回放
  - 重要性采样权重
  - `SumTree` 采样
- 统一输入范式
  - 向量观测直接输入 MLP
  - 离散状态可转成 one-hot 向量后输入 MLP
- 工程能力
  - 单次训练入口
  - 批量实验 manifest
  - 训练日志、评估日志、训练曲线
  - 模型保存与加载
  - 汇总结果导出为 CSV / JSON / Markdown / 对比图

## 安装

```bash
uv sync
```

或者：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 快速开始

### 1. 单次训练

直接运行：

```bash
python train.py
```

默认配置定义在 [config.py](config.py) 中。

当前默认环境是 `CartPole-v1`，这样可以先用最低成本验证训练链路。

几个常用配置项：

- `core.env_name`：环境名
- `core.env_family`：环境类型（`auto` / `vector` / `discrete` / `custom`）
- `training.hidden_sizes`：MLP 隐藏层结构
- `replay.use_per`：是否启用 PER
- `replay.obs_encoding`：观测编码方式（如 `identity` / `one_hot`）
- `core.max_steps`：总环境步数上限
- `replay.initial_random_steps`：纯随机 warmup 步数
- `replay.training_start_steps`：训练启动门槛
- `training.train_freq` / `training.gradient_steps`：训练频率
- `evaluation.eval_interval_env_steps` / `evaluation.eval_episodes`：评估节奏
- `evaluation.eval_epsilon` / `evaluation.eval_max_episode_steps`：评估时探索率与单局步数上限
- `evaluation.success_threshold`：成功率统计阈值

## 当前推荐环境设置

### `CartPole-v1`
用途：基础算法验证

建议：
- `env_family="vector"`
- `obs_encoding="identity"`
- 用于先验证 DQN / PER-DQN 链路是否跑通

### `Taxi-v3`
用途：主要策略决策实验

建议：
- `env_family="discrete"`
- `obs_encoding="one_hot"`
- 适合做动作选择、策略稳定性与关键决策分析

### `MountainCar-v0`
用途：补充稀疏奖励实验

建议：
- `env_family="vector"`
- `obs_encoding="identity"`
- 适合观察 PER 在稀疏奖励和延迟回报任务中的表现

## 2. 模型演示

```bash
python play.py
```

播放行为会读取 [config.py](config.py) 中的当前环境与模型配置。

如果默认模型文件不存在，脚本会明确提示当前将使用未训练策略，而不是静默假装加载成功。

## 3. 运行批量实验

实验入口在 [experiment.py](experiment.py)。

当前实验配置分成两层：

- `base_config`：单次训练配置
- `ExperimentSettings`：实验调度配置

可修改的常见项包括：

- `base_config.core.env_name`
- `base_config.core.max_steps`
- `base_config.training.learning_rate`
- `base_config.replay.use_per`
- `envs`
- `seeds`
- `variants`
- `output_root`

运行方式：

```bash
python experiment.py
```

默认目录结构示例：

```text
experiments/
  experiment_manifest.json
  CartPole-v1/
    dqn/
      seed_42/
        logs/
        models/
    per_dqn/
      seed_42/
        logs/
        models/
```

## 4. 汇总实验结果

在 [summarize_experiments.py](summarize_experiments.py) 中设置：

- `manifest_path`
- `output_dir`

然后运行：

```bash
python summarize_experiments.py
```

输出内容包括：

- `aggregate_results.csv`
- `aggregate_results.json`
- `aggregate_results.md`
- 每个环境一张对比曲线图

## 训练与日志产物

每次训练会在对应 `log_dir` 和 `save_dir` 下生成。

- 单次直接运行 `python train.py` 时，默认目录通常是 `runs/` 与 `models/`
- 通过 `python experiment.py` 运行批量实验时，每个 run 的目录通常是 `.../logs/` 与 `.../models/`

### `logs/`

- `config.json`：本次训练实际使用的配置
- `metrics.json`：完整日志，包含：
  - `episode_rewards`
  - `episode_lengths`
  - `episode_losses`
  - `epsilons`
  - `avg_rewards`
  - `eval_steps`
  - `eval_rewards`
  - `global_steps`
  - `step_losses`
  - `step_epsilons`
- `run_summary.json`：用于实验汇总的摘要结果
- `training_curves.png`：训练曲线图

### `models/`

- `*_ep{n}.pth`：按 `save_freq` 定期保存的模型文件

这些文件的定位是**模型保存与结果归档**：它们保存模型权重与必要元数据，用于评估、播放和实验结果留档。

当前版本不以恢复训练为目标，因此这里的保存文件应理解为“可加载模型做评估/回放的模型工件”，而不是“可无缝续训的训练快照”。

## 当前实现说明

### 1. 网络结构

当前默认网络是 MLP，而不是 CNN。

- 向量状态：直接输入 MLP
- 离散状态：先编码为向量，再输入 MLP

这让项目更适合：
- 经典控制环境
- 离散决策环境
- 自定义特征状态小游戏环境

### 2. ReplayBuffer 设计

当前 replay buffer 直接存储完整：

- `state`
- `action`
- `reward`
- `next_state`
- `terminated`
- `truncated`

因此不再依赖图像帧重建，也不再假设存在帧堆叠。

### 3. PER 实现

当前 `PER-DQN` 使用：

- `SumTree` 做优先级采样
- importance sampling weights 做偏差修正
- TD error 更新优先级

## 论文或实验汇报建议

建议把**评估指标**作为主结论来源，把训练期指标作为辅助分析来源。

建议优先使用以下指标：

- `final_eval_reward`
- 固定训练步数下的 `eval_reward`
- 多 seed 的均值与标准差
- `best_eval_reward`
- `final_eval_success_rate`（当环境设置了成功阈值时）

在写毕业论文时，建议：

- `CartPole-v1` 用于基础验证
- `Taxi-v3` 作为策略决策分析主体
- `MountainCar-v0` 作为补充实验

如果后续加入谷歌小恐龙环境，建议继续使用**特征工程状态表示**，这样更容易控制训练成本，也更适合论文分析。

## 主要文件说明

- [train.py](train.py)：单次训练入口，包含训练调度、评估和定期模型保存逻辑
- [play.py](play.py)：模型演示
- [experiment.py](experiment.py)：批量实验入口
- [summarize_experiments.py](summarize_experiments.py)：结果汇总与绘图
- [config.py](config.py)：训练配置定义
- [dqn/agent.py](dqn/agent.py)：DQN / PER 训练逻辑
- [dqn/replay_buffer.py](dqn/replay_buffer.py)：通用回放缓冲区与 `SumTree`
- [dqn/network.py](dqn/network.py)：MLP Q 网络
- [dqn/env.py](dqn/env.py)：Gym 环境构造与观测编码
- [dqn/utils.py](dqn/utils.py)：日志、绘图与随机种子工具

## 当前限制

- 当前仓库已经完成从 Atari 风格到低维状态风格的核心迁移，但不同环境的默认超参数仍需进一步打磨。
- 当前 PER 在短程 smoke test 中已可运行，但要用于论文主结论前，仍建议做更稳的多 seed 验证。
- 仓库目前还没有完整的自动化测试体系，当前验证主要依赖模块测试与短程训练测试。
- 论文所需的“策略决策分析日志”还未在这一轮中实现，后续会作为下一阶段补充。

## 参考文献

- Mnih et al., 2015. Human-level control through deep reinforcement learning.
- Schaul et al., 2016. Prioritized Experience Replay.

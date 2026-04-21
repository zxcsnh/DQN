# Atari 上的 Nature DQN 与 PER-DQN 对比项目

这是一个基于 PyTorch 的 Atari 强化学习项目，包含单次训练、模型回放、批量实验和结果汇总。当前实现已经从最初的基础 DQN 演进为更接近可复现实验基线的版本：

- 标准分支使用更接近 Nature 2015 的原版 `DQN` target
- PER 分支在此基础上加入 `Prioritized Experience Replay`
- 训练主循环支持 warmup、固定训练频率、评估环境复用和模型保存
- 回放缓冲区已重构为按帧去重存储，PER 采样使用 `SumTree`

适合做课程实验、算法对比和中小规模论文复现实验。

## 当前功能

- `DQN`：目标网络、experience replay、epsilon-greedy 探索
- `PER-DQN`：优先经验回放、重要性采样权重、`SumTree` 采样
- Atari 预处理：灰度化、跳帧、帧堆叠、奖励裁剪、可选生命丢失终止
- 训练优化：
  - `initial_random_steps` 前纯随机探索，之后从 `epsilon=1.0` 开始 epsilon-greedy
  - `training_start_steps`、`train_freq`、`gradient_steps` 共同控制训练节奏
  - 评估环境复用，减少重复初始化开销
  - step 级训练日志
  - 按 `save_freq` 定期保存模型文件，供后续评估、回放或结果留档
- 工程能力：
  - 模型工件保存/加载（用于评估、演示与结果归档）
  - 批量实验 manifest
  - 以评估指标为主的 CSV / JSON / Markdown 汇总与对比图输出

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

训练配置在 [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:1) 中定义。

几个常用配置项：

- `env_name`：环境名
- `use_per`：是否启用 PER
- `max_steps`：总环境步数上限
- `initial_random_steps`：纯随机 warmup 步数
- `training_start_steps`：训练启动门槛
- `train_freq` / `gradient_steps`：训练频率
- `eval_interval_env_steps` / `eval_episodes`：评估节奏
- `eval_epsilon` / `eval_max_episode_steps`：评估时的探索率与单局步数上限

需要特别注意：这个项目的训练协议和评估协议是**有意不对称**的。
- 训练环境默认启用 reward clipping，并且默认启用 `terminal_on_life_loss`
- 评估环境固定关闭 reward clipping，并关闭 `terminal_on_life_loss`
- 因此训练期 reward（例如 `final_train_avg_reward_100`）与评估 reward（例如 `final_eval_reward`）不是同一统计口径，不能直接横向比较

如果做实验汇报，建议优先以评估指标为主，把训练期指标当作辅助观察信号。

还需要注意两层不同语义：
- `terminated` / `truncated` 会共同决定一个 episode 是否结束
- 但当前训练 target 的 bootstrap mask **只使用 `terminated`**，`truncated` 不会直接把 Bellman target 置零

可以把三套协议理解为：
- train：可启用 reward clipping，可启用 `terminal_on_life_loss`，用于优化训练
- eval：关闭 reward clipping，关闭 `terminal_on_life_loss`，使用 `eval_epsilon` 和 `eval_max_episode_steps` 做性能评估
- play：沿用 eval 协议，用于回放和录制视频

因此，episode 停止条件与 target mask 条件不是同一个概念；训练 reward、评估 reward、回放表现也不应混为同一口径。

如果默认模型文件不存在，`python play.py` 会明确提示当前将使用未训练策略，而不是静默假装加载成功。

### 2. 模型演示或录制视频

```bash
python play.py
```

是否渲染、是否录制视频等设置在 [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:1) 中修改。

### 3. 运行批量实验

实验入口在 [experiment.py](/e:/Programming_Language/MachineLearning/RL/DQN/experiment.py:1)。

当前实验配置分成两层：

- `base_config`：单次训练配置
- `ExperimentSettings`：实验调度配置

可修改的常见项包括：

- `base_config.env_name`
- `base_config.max_steps`
- `base_config.learning_rate`
- `base_config.use_per`
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
  ALE_Pong-v5/
    dqn/
      seed_42/
        logs/
        models/
    per_dqn/
      seed_42/
        logs/
        models/
```

### 4. 汇总实验结果

在 [summarize_experiments.py](/e:/Programming_Language/MachineLearning/RL/DQN/summarize_experiments.py:1) 中设置：

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

下面以批量实验的默认目录结构为例说明：

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
- `run_summary.json` 中包含：
  - `best_eval_reward`
  - `best_eval_step`
  - `final_eval_reward`
  - `final_train_avg_reward_100`
- `training_curves.png`：训练曲线图

### `models/`

- `*_ep{n}.pth`：按 `save_freq` 定期保存的模型文件

这些文件的定位是**模型保存与结果归档**：它们保存模型权重与必要元数据，用于评估、播放、视频录制和实验结果留档。

当前版本**不以恢复训练为目标**，因此这里的保存文件应理解为“可加载模型做评估/回放的模型工件”，而不是“可无缝续训的训练快照”。

## 当前实现细节说明

### 1. 标准分支已回到 Nature DQN 结构

当前“普通分支”使用与 Nature 2015 更接近的标准 DQN target、Nature 风格的 RMSProp 配置，以及按 frame budget 驱动的训练协议。

也就是说：

- `dqn` 分支：`Nature DQN`
- `per` 分支：`Nature DQN + PER`

### 2. ReplayBuffer 已做结构重构

当前 replay buffer 不再直接存完整 stacked `state / next_state`，而是：

- 按帧去重存储
- 按采样索引重建 frame stack

这样可以显著降低回放缓冲区的内存压力，也让更大的 buffer 更可行。

### 3. PER 已使用 SumTree

当前 PER 采样不再依赖每次全量概率归一化加 `np.random.choice(..., p=probs)`，而是改为 `SumTree` 结构采样。

## 论文或实验汇报建议

建议把**评估指标**作为主结论来源，把**训练期指标**作为辅助分析来源。

建议优先使用以下评估指标：

- `final_eval_reward`
- 固定训练步数下的 `eval_reward`
- 多 seed 的均值与标准差
- 不同环境下的评估曲线
- `best_eval_reward`（适合作为峰值表现展示，不建议单独作为最终结论）

建议把 `final_train_avg_reward_100` 这类训练期指标作为辅助分析项，而不是最终结论的唯一依据。特别是在启用了 reward clipping 时，训练期 reward 和真实评估分数的含义并不完全一致。

## 主要文件说明

- [train.py](/e:/Programming_Language/MachineLearning/RL/DQN/train.py:1)：单次训练入口，包含训练调度、评估和定期模型保存逻辑
- [play.py](/e:/Programming_Language/MachineLearning/RL/DQN/play.py:1)：模型演示与视频录制
- [experiment.py](/e:/Programming_Language/MachineLearning/RL/DQN/experiment.py:1)：批量实验入口
- [summarize_experiments.py](/e:/Programming_Language/MachineLearning/RL/DQN/summarize_experiments.py:1)：结果汇总与绘图
- [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:1)：训练配置定义
- [dqn/agent.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/agent.py:1)：Nature DQN / PER 训练逻辑
- [dqn/replay_buffer.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/replay_buffer.py:1)：回放缓冲区与 `SumTree`
- [dqn/network.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/network.py:1)：卷积 Q 网络
- [dqn/env.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/env.py:1)：Atari 环境预处理
- [dqn/utils.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/utils.py:1)：日志、绘图与随机种子工具

## 当前限制

- 当前网络已经支持按 `frame_size` 动态推导卷积输出尺寸，但回放或播放旧模型文件时，如果模型不是当前版本保存的，仍建议优先使用默认 `84x84`。
- `record_video()` 仍然会先缓存整局帧再写视频，长 episode 下会带来额外内存占用。
- 仓库目前还没有完整的自动化测试体系，重构后的验证主要依赖 smoke test 和短程训练测试。

## 参考文献

- Mnih et al., 2015. Human-level control through deep reinforcement learning.
- Schaul et al., 2016. Prioritized Experience Replay.

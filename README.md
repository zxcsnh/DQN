# Atari 上的 DQN 与 PER-DQN 对比项目

这是一个基于 PyTorch 的 Atari DQN 项目，支持普通 DQN 与优先经验回放 DQN 的训练、评估、批量对比实验和结果汇总，适合做课程实验或论文实验。

## 功能简介

- 普通 DQN：包含经验回放、目标网络、epsilon-greedy 探索
- PER-DQN：支持优先经验回放与重要性采样权重
- Atari 预处理：帧堆叠、跳帧、奖励裁剪、可选生命丢失终止
- 单次训练与模型回放
- 多环境、多随机种子的批量实验
- 自动汇总结果，生成 CSV、JSON、Markdown 和对比图

## 安装方式

```bash
uv sync
```

或者：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 基本使用

### 1. 单次训练

直接运行：

```bash
python train.py
```

训练参数在 [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:1) 中修改。

### 2. 模型演示或录制视频

```bash
python play.py
```

是否渲染、是否录制视频等设置，也在 [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:1) 中修改。

### 3. 运行论文对比实验

直接打开 [experiment.py](/e:/Programming_Language/MachineLearning/RL/DQN/experiment.py:1)，修改 `ExperimentSettings` 中的固定配置，例如：

- `envs`
- `seeds`
- `variants`
- `max_steps`
- `eval_freq`
- `eval_episodes`

然后运行：

```bash
python experiment.py
```

默认会在 `experiments/` 下生成类似这样的目录结构：

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

直接打开 [summarize_experiments.py](/e:/Programming_Language/MachineLearning/RL/DQN/summarize_experiments.py:1)，修改 `SummarySettings` 中的固定配置：

- `manifest_path`
- `output_dir`

然后运行：

```bash
python summarize_experiments.py
```

汇总结果会输出到 `summary/` 目录，包含：

- `aggregate_results.csv`
- `aggregate_results.json`
- `aggregate_results.md`
- 每个环境一张对比曲线图

## 论文实验推荐流程

1. 在 `experiment.py` 中设置 2 到 4 个 Atari 环境。
2. 为每种方法设置 3 到 5 个随机种子。
3. 保持 DQN 和 PER-DQN 除 `use_per` 外的其他超参数一致。
4. 跑完实验后执行 `summarize_experiments.py`。
5. 在论文中报告均值、标准差和对比图。

## 主要文件说明

- [train.py](/e:/Programming_Language/MachineLearning/RL/DQN/train.py:1)：单次训练入口
- [play.py](/e:/Programming_Language/MachineLearning/RL/DQN/play.py:1)：模型演示与视频录制
- [experiment.py](/e:/Programming_Language/MachineLearning/RL/DQN/experiment.py:1)：批量实验入口
- [summarize_experiments.py](/e:/Programming_Language/MachineLearning/RL/DQN/summarize_experiments.py:1)：结果汇总与绘图
- [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:1)：训练基础配置
- [dqn/agent.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/agent.py:1)：DQN 与 PER 训练逻辑
- [dqn/replay_buffer.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/replay_buffer.py:1)：普通经验回放与优先经验回放
- [dqn/env.py](/e:/Programming_Language/MachineLearning/RL/DQN/dqn/env.py:1)：Atari 环境预处理

## 训练产物说明

每次运行会在对应 `logs/` 目录下保存：

- `config.json`：本次运行实际使用的配置
- `metrics.json`：训练过程中的回报、损失与评估曲线
- `run_summary.json`：供批量统计使用的摘要结果
- `training_curves.png`：单次训练曲线图

模型文件保存在对应的 `models/` 目录下。

## 写论文时建议使用的指标

- `best_eval_reward`
- `final_avg_reward_100`
- 多随机种子的均值与标准差
- 不同环境下的收敛曲线

建议以评估回报作为主要比较指标，不要直接用训练时的原始 reward 作为最终结论。

## 参考文献

- Mnih et al., 2015. Human-level control through deep reinforcement learning.
- Schaul et al., 2016. Prioritized Experience Replay.

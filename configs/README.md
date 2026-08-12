# 四环境复现配置

配置只保留两类：

```text
configs/
├── envs/            # 每个环境的环境、数据和评测配置
└── methods/         # 每个方法的模型和训练配置
```

训练时直接选择一个环境配置和一个方法配置进行组合，不再维护额外的矩阵文件。
完整实验仍然是 4 个环境 × 7 个方法。当前只执行一个训练种子；首个 LeWM PushT
运行沿用已记录诊断实验的 `seed=3072`。只有用户明确决定扩大统计评测后才增加其他
种子。训练入口采用明确参数，例如 `--env pusht --method lewm --seed 3072`。

## 当前矩阵状态

| 方法 | PushT | Cube | Reacher | Two-Room |
| --- | --- | --- | --- | --- |
| LeWM | ready | ready | ready | ready |
| PLDM | ready | ready | ready | ready |
| DINO-WM（pixels-only） | ready | ready | ready | ready |
| GCBC | ready | ready | ready-new | ready |
| GCIVL | ready | ready | ready-new | ready |
| GCIQL | ready | ready | ready-new | ready |
| TD-MPC2 | protocol-gated | protocol-gated | protocol-gated | protocol-gated |

`ready-new` 表示训练配置完整，但论文没有发布该方法在 Reacher 上的 checkpoint
或数值，因此这是新的受控复现，不是已有论文结果的重复运行。

TD-MPC2 被显式门控，是因为它需要 reward/Q 学习，而 LeWM 四环境主协议是离线、
reward-free、任意目标条件控制。没有先确定 goal-conditioned reward relabeling 或独立
在线训练协议，直接启动 TD-MPC2 会产生不可比较的结果。其四个运行单元仍保留在
研究范围中，但训练入口必须读取 `methods/tdmpc2.yaml` 的协议门控并拒绝默认启动。

## 采用的协议

- 固定依赖：`stable-worldmodel[all]==0.1.1`；
- 主结果使用相同离线数据、训练/验证划分、评测 episode、起点、目标偏移和预算；
- 当前训练随机种子：`3072`（单 seed）；
- 数据按 episode 使用固定种子 42 划分，不能让不同方法重新随机划分；
- 每次评测 50 条轨迹，目标偏移 25 个环境步，执行预算 50 个环境步；
- world model 统一使用 CEM：300 candidates、30 elites、horizon 5、action block 5；
- PushT 使用 30 次 CEM 迭代，其他环境使用 10 次；
- world model 训练 10 epochs，符合 LeWM 论文附录中的四环境协议；
- policy baseline 使用其发布配置的训练预算：GCBC/GCIVL/GCIQL 为 100 epochs；
- 主要 DINO-WM 结果不使用 proprioception；`DINO-WM+prop` 应作为单独消融，不能
  覆盖 pixels-only 主结果。

## 论文与发布配置的差异

LeWM 论文明确写明四个 world model 均训练 10 epochs；公开仓库的通用训练 YAML
当前默认 100 epochs。本套件使用论文值 10。LeWM 的 SIGReg 权重采用发布配置中的
`0.09`；论文正文将默认值表述为约 `0.1`。这两个来源都记录在 method 配置的
`provenance` 字段中。

## 数据放置

官方 Hugging Face 数据是 `.zst` 压缩归档，`stable-worldmodel==0.1.1` 不会直接
读取这些压缩包。先解压，再把得到的 HDF5 文件放到
`$STABLEWM_HOME/datasets/` 下，保持对应 `envs/*.yaml` 中记录的相对名称。

当前仓库已有 PushT 压缩包：

```text
data/lewm-pusht/pusht_expert_train.h5.zst
```

不要把数据集、checkpoint 或训练输出提交到 Git。其余三个官方压缩包合计很大，
正式下载和解压前应先确认 GPU 机器的磁盘空间。

## 执行边界

这些文件现在是实验配置的唯一事实来源。`scripts/train.py` 已实现 PushT 上的
LeWM、PLDM、DINO-WM、GCBC、GCIVL 和 GCIQL；其他环境的训练适配仍未实现。
`scripts/evaluate.py` 已按环境配置调用统一的 dataset-backed `world.evaluate(...)`
协议，但每个环境仍需要匹配的数据集和可加载 checkpoint。GCIVL/GCIQL 分别保留
价值（或 V/Q）和策略两个阶段。训练入口通过
`import stable_worldmodel as swm` 使用已安装的 `0.1.1`，读取这里的配置进行轻量
组装，不复制或修改上游训练脚本。

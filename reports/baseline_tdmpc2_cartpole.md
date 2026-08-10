# TD-MPC2 CartPole Baseline 复现记录

记录日期：2026-08-09

## 目的

先跑通一个 `stable-worldmodel` 官方 baseline，确认 world model 任务的环境、
训练、规划、checkpoint 和评测链路，并观察任务是否可学。

本次选择在线 TD-MPC2 和 DMControl CartPole。它不代表后续论文最终使用的任务，
但适合作为第一个低维、计算成本可控的端到端 baseline。

## 运行环境

- 平台：Gemini 开发环境
- `stable-worldmodel`：`0.1.1`
- 上游源码快照：`67017b79ef194e96fd96d201fa3ba51ffff62775`
- Python：3.11.8
- PyTorch：2.2.2+cu121
- CUDA：12.1
- GPU：`B1.gpu.small`，可见显存约 5950 MiB
- 持久化工作区：`/gemini/code/stable-worldmodel/workspace`

环境验证已通过，包括 CUDA、Stable World Model 导入和 64×64 环境观测。

## 任务与配置

- 环境：`swm/CartpoleDMControl-v0`
- 任务：CartPole swing-up/balance
- observation dimension：5
- action dimension：1
- episode length：500 environment steps
- 训练步数：100,000
- 随机种子：1
- TD-MPC2 planning horizon：3
- discount：0.99
- seed steps：5,000
- batch size：256
- training CEM samples：256
- training CEM iterations：4
- evaluation frequency：每 10,000 steps
- evaluation episodes：3

TD-MPC2 联合学习 observation encoder、latent dynamics、reward model 和
Q-ensemble，并使用 CEM 在学习到的 latent world model 中规划动作。

## 发现的任务差异

`stable-worldmodel` 该源码快照中的官方 `CartpoleDMControlWrapper` 使用：

```python
cartpole.Balance(swing_up=True, sparse=True, random=seed)
```

因此官方任务是稀疏奖励 swing-up。为了判断算法链路是否正常，另运行了只将
`sparse=True` 改为 `sparse=False` 的标准 DMControl dense reward 版本。模型、
优化器、batch、CEM 和训练步数保持一致；此外只有 headless rendering、评测频率
和日志频率等平台适配差异。

## 结果

### 官方稀疏奖励版本

| Step | Mean evaluation reward（3 episodes） |
| ---: | ---: |
| 10k | 0.00 |
| 20k | 0.00 |
| 30k | 0.00 |
| 40k | 0.00 |
| 50k | 0.00 |
| 60k | 0.00 |
| 70k | 0.00 |
| 80k | 0.00 |
| 90k | 4.67 |
| 100k | 0.00 |

该配置在 100k steps 内没有稳定学会任务，最佳周期评测为 4.67。

### Dense reward 诊断版本

| Step | Mean evaluation reward（3 episodes） |
| ---: | ---: |
| 10k | 371.54 |
| 20k | 770.54 |
| 30k | 832.31 |
| 40k | 833.11 |
| 50k | 835.32 |
| 60k | 863.23 |
| 70k | 857.97 |
| 80k | 820.82 |
| 90k | 853.58 |
| 100k | 864.55 |

训练完成后，从 `best_model.pt` 重新加载并独立评测 3 个 episodes：

```text
mean_reward = 864.4722
```

独立结果与训练过程的 864.55 一致，说明 checkpoint 保存、加载、CEM 规划和
评测链路可以复现。

## 当前结论

1. Stable World Model 的基础环境、TD-MPC2、CUDA 训练、CEM 规划和 checkpoint
   链路已经跑通。
2. CartPole dense swing-up 对当前 TD-MPC2 配置是明显可学的，约 20k–30k steps
   已进入较高回报区间。
3. 官方 sparse reward 版本在 seed 1、100k steps 下存在严重探索瓶颈。这个结果
   不能直接解释为 TD-MPC2 实现失效，因为相同模型在 dense 版本上能够稳定学习。
4. Dense 版本是诊断实验，不是与官方 sparse 任务完全相同的复现。后续报告必须
   明确标注 reward mode，不能混合比较两种任务的数值。

## 局限

- 目前只有一个训练随机种子。
- 每次周期评测及独立评测只有 3 个 episodes。
- 成功结果修改了 reward density，因此不能声称复现了官方 sparse 任务性能。
- 当前约 6GB vGPU 足够运行这个低维 baseline，但未验证像素型 LeWM 的正式训练。

## 下一步建议

1. 将 dense CartPole 固定为工程 smoke baseline，用于快速验证训练和评测链路。
2. 至少补 seed 2、3，报告均值和标准差。
3. 如果研究重点是像素 world model，再进入 PushT + LeWM/DINO-WM/PLDM；在正式
   训练前先核对数据集大小、显存需求和官方 checkpoint。
4. 最终方法必须在同一 reward mode、数据、步数、solver 和评测预算下与 baseline
   比较。

## Artifact 状态

原始训练日志、模型和 checkpoints 已在云端清理，不进入 GitHub。本报告仅保留轻量
配置、汇总指标和局限说明。由于原始逐 episode 结果已经不可用，这组结果不能用于
后续正式统计检验；需要正式比较时，应使用当前统一入口重新运行并保存受控的外部
artifact。

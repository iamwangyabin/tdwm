# Aligned E2E MC-GT-LeWM Cube 正式 O50 结果

正式评测完成：2026-08-27（Asia/Shanghai）

记录状态：预先锁定 epoch 10 checkpoint、50 个 episode、O50 start--goal offset 和完整
CEM 预算后的单训练 seed 正式评测。结果是一个正向观察，但不足以声称方法在统计意义上
优于 baseline。

## 核心结果

| 方法 | 成功数 | success rate | 评测耗时 |
| --- | ---: | ---: | ---: |
| Aligned E2E MC-GT-LeWM | **31 / 50** | **62%** | 332.36 s |
| 配对 LeWM `lewm_seed3072_e10_matched_o50_retry` | 27 / 50 | 54% | 379.93 s |

两项评测使用完全相同的 50 个 start--goal pair。观测差为 `+8` 个百分点，即 Aligned
净增 4 个成功 episode。

## 配对统计

| 配对结果 | episode 数 |
| --- | ---: |
| 两者都成功 | 26 |
| 仅 Aligned 成功 | 5 |
| 仅 LeWM 成功 | 1 |
| 两者都失败 | 18 |

- 双侧 exact McNemar：`p=0.21875`。
- 100,000 次 episode-level paired bootstrap、固定随机种子 `20260827`：success-rate
  difference 的 percentile 95% interval 为 `[0, +18]` 个百分点。

因此准确表述是：**本次锁定的单 seed、50-episode 评测观测到 Aligned 比配对 LeWM 高
8 个百分点，但差异没有达到常用统计显著性标准，也没有跨训练 seed 验证。**

## 锁定评测协议

| 项目 | 值 |
| --- | --- |
| 环境 | `swm/OGBCube-v0`，state observation，成功阈值 0.04 m |
| 数据集 | `quentinll/lewm-cube` Lance；10,000 episodes，2,010,000 transitions |
| 评测 | 50 episodes；goal offset 50；同一数据 episode 内选择 start/goal |
| episode 选择 | Stable World Model 0.1.1 valid-row sampler；无放回；与 LeWM 完全相同 |
| CEM | horizon 5；300 candidates；30 iterations；30 elites；initial variance 1.0 |
| 执行方式 | action block 5；每 5 个环境 step 重规划；episode budget 100 |
| planning seed | 42 |
| 数值精度 | FP32 |

正式配置为
[`configs/experiment/aligned_e2e_mc_gt_lewm_cube_seed3072_o50.yaml`](../configs/experiment/aligned_e2e_mc_gt_lewm_cube_seed3072_o50.yaml)。
该配置在正式运行前由提交 `831c9de` 锁定并通过协议测试；正式结果记录的完整评测 revision
为 `831c9de8e3e03919309af43f480a42bc88afc46e`。

## Checkpoint 与运行来源

| 项目 | 值 |
| --- | --- |
| training seed | 3072 |
| selected checkpoint | epoch 10，127,960 optimizer updates |
| base world-model parameters | 18,034,628 |
| base checkpoint SHA-256 | `31c5ce04e9b5eec66e1015aa4d8318ccac2d4fa980a2f49937e70db56c653e72` |
| value parameters | 682,497 |
| joint/value checkpoint SHA-256 | `1912e444000888ef0beb1a8a0fd902292f7c5caf7bbb384500487526602adcea` |
| training revision | `b9a33a75188d7971ac9225c9078624c580c8c68f` |
| Stable World Model | `stable-worldmodel[all]==0.1.1` |
| runtime | Python 3.11.15；PyTorch 2.13.0+cu130；NVIDIA GeForce RTX 3090 |

## Artifact 审计

正式轻量结果保存在本地被 Git 忽略的目录：

```text
outputs/server_experiments/3090/outputs/aligned_e2e_mc_gt_lewm_cube_seed3072_epoch10_o50/
```

服务器原始目录为：

```text
/home/yabin/tdwm/outputs/aligned_e2e_mc_gt_lewm_cube_seed3072_epoch10_o50/
```

关键文件校验值：

| 文件 | SHA-256 |
| --- | --- |
| `results.json` | `811ab645e9ffeaeff201cbbd644c3ae3808906740f042bd6d3a2215f1041fa1d` |
| `protocol_manifest.json` | `75a71fab8dda309c449d7a52d579f77bf76ad83c07561f2c5f0fc107e1c9086a` |
| `evaluation.log` | `3a6f90fcef335e32587ab9e25cd6549ad595182e0f532e29ef3ce38dd0c0f057` |

checkpoint、原始日志和逐 episode 原始 artifact 不提交到 GitHub；GitHub 只保存锁定配置、
测试和本轻量审计报告。

## 结论边界与下一步

本结果支持继续开展 Aligned E2E MC-GT-LeWM 的受控复现，因为它在预先锁定的 O50 协议
上产生了正向差异。它不支持“已优于 LeWM”的结论，原因包括：

1. 只有一个训练 seed 和一个 planning seed；
2. 只有一组 50 个配对 episode；
3. exact McNemar `p=0.21875`；
4. Aligned 与 baseline 使用各自训练的 world-model checkpoint，不能把差异解释成单独
   添加 tail value 的纯消融效应。

下一步应在预先固定同一评测协议后增加独立训练 seeds，并报告每个 seed 的逐 episode
配对结果、汇总置信区间、训练/规划成本和 checkpoint provenance。

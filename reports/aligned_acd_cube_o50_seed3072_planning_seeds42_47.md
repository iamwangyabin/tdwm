# Aligned A/C/D Cube O50：六组 planning selections 的配对归档

归档日期：2026-08-27（Asia/Shanghai）

本报告保留固定 training seed 3072、planning-selection seeds 42--47 的 300 个
配对 Cube O50 episode。它不覆盖 planning seed 42 的旧单次报告
[`aligned_e2e_mc_gt_lewm_cube_seed3072.md`](aligned_e2e_mc_gt_lewm_cube_seed3072.md)；
旧报告记录当时真实的 `62%` 对 `54%` 观察，本报告记录后续消融如何改变了核心解释。

## 因果单元定义

| 标签 | World model | 推理 cost | 状态 |
| --- | --- | --- | --- |
| A | Original LeWM | terminal only | 已完成 |
| B | Original LeWM | matched anchored tail | **尚未完成** |
| B-prime | Original LeWM | Aligned-coordinate tail | 坐标失配诊断，不是 B |
| C | Aligned world | terminal only | 已完成 |
| D | Aligned world | matched anchored tail | 已完成 |

因此，本报告可以干净估计 `C-A` 和 `D-C`，但还不能计算原始坐标中的 matched tail
效应 `B-A`，也不能计算完整 interaction `(D-C)-(B-A)`。

## 协议与范围

- 环境：`swm/OGBCube-v0`，goal offset 50。
- 数据：同一份 10,000-episode、2,010,000-transition JPEG-100 Lance 数据集。
- 训练 seed：固定为 3072；没有训练六个独立模型。
- 评测 selections：planning seeds 42、43、44、45、46、47，各 50 episodes。
- 每个 planning seed 内，A/C/D 的 episode、start step、goal step 和 selection 文件
  SHA-256 完全一致。
- CEM：horizon 5、300 candidates、30 iterations、30 elites、action block 5，
  每 5 个环境 step 重规划，episode budget 100。
- seeds 42、43、45、47 在 RTX 3090 运行；44、46 在云端 RTX 4080 运行。每个
  planning seed 的三个 paired cells 均在同一机器和 runtime 下运行。

## 六组完整结果

| Planning selection seed | A | C | D |
| ---: | ---: | ---: | ---: |
| 42 | 54% (27/50) | 56% (28/50) | 62% (31/50) |
| 43 | 66% (33/50) | 74% (37/50) | 70% (35/50) |
| 44 | 44% (22/50) | 46% (23/50) | 46% (23/50) |
| 45 | 46% (23/50) | 54% (27/50) | 52% (26/50) |
| 46 | 48% (24/50) | 52% (26/50) | 58% (29/50) |
| 47 | 48% (24/50) | 54% (27/50) | 46% (23/50) |
| **300 episodes** | **51.0% (153/300)** | **56.0% (168/300)** | **55.67% (167/300)** |

描述性方向同样重要：`C-A` 在 6/6 组 selections 中均为正；`D-C` 为 2 组提升、
1 组持平、3 组下降。Aligned world-only 的方向跨 selections 一致，而 inference-time
tail 的影响不稳定。

## 配对统计

| Contrast | both | left only | right only | neither | 差值 | exact McNemar p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A -> C | 142 | 11 | 26 | 121 | +5.00 pp | 0.02007 | 0.06022 |
| C -> D | 154 | 14 | 13 | 119 | -0.33 pp | 1.00000 | 1.00000 |
| A -> D | 139 | 14 | 28 | 119 | +4.67 pp | 0.04356 | 0.08712 |

这里把三个 contrasts 作为平级比较并报告 Holm correction。A -> C 没有在实验前注册为
primary contrast，因此不能事后只用未校正 `p=0.02007` 声称显著结果。正确表述是：

> 在固定 training seed 3072 的 checkpoint 上，跨六组 matched planning selections、
> 共 300 episodes，Aligned world-only 比 Original LeWM world-only 高 5.0 个百分点；
> 这是一个方向一致的配对观察，但不是多 training-seed 优势证明。

## 推理 tail 的准确解释

`D-C=-0.33 pp`，即 300 个 episode 中 D 比 C 少成功 1 个。当前 anchored tail 在推理
阶段没有观测到总体 success-rate 收益，但不能说它没有参与规划。O5 机制诊断得到：

- terminal-total ranking correlation：`0.92978`；
- tail/terminal 平均比：`0.35704`；
- 最终 CEM iteration 中 tail 改变最佳 candidate：`27.42%`；
- boundary value absolute maximum：`0.0`。

因此准确结论是：**tail 确实重排了约四分之一的最终 candidate selection，但这种重排
没有转化为 pooled success-rate 增益。** O5 只用于机制诊断，不能用其 2/5 成功率作
性能结论；candidate predicted cost 与真实反事实 rollout outcome 的相关性尚未采集。

## Invalid cross-coordinate diagnostic：B-prime

B-prime 将 Original LeWM world model 与在 Aligned latent coordinates 中训练的 tail
组合，结果为 `46%`（23/50）。相同 seed-42 selection 上 A 为 `54%`，但该 `-8 pp`
不能解释为 matched tail effect，因为 world 和 value 不在同一 latent coordinate system。
它只诊断了跨坐标直接复用 tail 的不兼容性。

现有 objective-version-1、非 anchored 的 Original-world MC head 为 `52%`（26/50），
同样不能代替 B。完整 `2x2` 设计仍缺在 Original LeWM coordinates 中重新训练的 matched
boundary-anchored tail。

## 样本重叠审计

- 300 个 `pair_hash` 全部唯一；跨 planning seeds 没有完全相同的 start--goal pair。
- 共有 296 个 unique source episodes。
- episode 5132、7002、8400、9267 各出现两次，但 start/goal steps 不同。

因此不能把 source episode 数写成 300，但本批次没有 exact pair duplication。

## O200 失败记录

原计划的单次 200-environment vectorized evaluation 在 RTX 3090 主机上构造 MuJoCo
环境时超过约 15 GiB host-memory 预算，被操作系统终止。随后执行的是六组独立、matched
的 O50 selections，总计 300 paired episodes。它们不得表述为一次 O200 execution。

## Checkpoint 与 provenance

| Artifact | SHA-256 |
| --- | --- |
| Original LeWM world | `0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579` |
| Aligned world | `31c5ce04e9b5eec66e1015aa4d8318ccac2d4fa980a2f49937e70db56c653e72` |
| Aligned anchored tail | `1912e444000888ef0beb1a8a0fd902292f7c5caf7bbb384500487526602adcea` |
| Existing unanchored Original-world MC head | `6ae24ffc715a761fd9a604564252d9414000d7661c175d7b6af2b1fe057d676f` |

所有运行均使用 `stable-worldmodel==0.1.1`。RTX 3090 运行记录 Python 3.11.15、
PyTorch 2.13.0+cu130；RTX 4080 运行记录 Python 3.12.3、PyTorch 2.8.0+cu128。
逐运行 Git revision、GPU、elapsed time、结果/selection/protocol SHA-256 和服务器原始路径
见机器可读的
[`selection_manifests.json`](artifacts/aligned_acd_o50_seed3072/selection_manifests.json)。

旧 evaluator 没有持久化 run-time dirty-worktree boolean，因此归档将该字段保存为
`null`，并明确记录缺失原因；没有事后猜测 clean/dirty 状态。

## 机器可读归档与复算

归档目录：
[`reports/artifacts/aligned_acd_o50_seed3072/`](artifacts/aligned_acd_o50_seed3072/README.md)。

复算命令：

```bash
python scripts/summarize_aligned_acd_o50.py --check
shasum -a 256 -c reports/artifacts/aligned_acd_o50_seed3072/checksums.sha256
```

## 当前结论与下一步

这批结果把项目当前最合理的解释从“推理时追加 tail 带来主要提升”改为：

> **当前观察到的主要提升来自 Aligned long-horizon supervision 下训练出的 world
> model；推理时追加 anchored tail 虽然会改变候选排序，但没有稳定总体增益。**

该结论只适用于一个 training seed。下一步必须完成 matched B、candidate-level 真实
反事实 rollout 诊断，并增加独立 training seeds；在这些实验完成前，不声称方法优于
baseline。

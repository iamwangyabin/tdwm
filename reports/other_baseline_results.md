# 其他 World Model 实验结果与复现优先级

记录日期：2026-08-09<br>
最后更新：2026-08-12

## 目的

TD-MPC2 CartPole 只用于确认环境、训练、规划和 checkpoint 链路能够工作，不能代表
本项目最终研究任务。本文整理 Stable World Model / LeWM 相关工作的公开实验结果，
区分论文结果、第三方复现和我们自己的服务器结果，并据此确定下一步 baseline。

本文只负责**数值来源与复现状态**，不再承担完整的创新性比较。RL-assisted LeWM 与
value-aware model learning、bisimulation、zero-shot RL、self-predictive RL 及 2026 年
直接 JEPA 改进的系统审计见
[`rl_assisted_lewm_related_work_review.md`](rl_assisted_lewm_related_work_review.md)。研究
提案见 [`rl_assisted_lewm_research_proposal.md`](rl_assisted_lewm_research_proposal.md)。

三类证据必须严格分开：

1. **文献机制比较**：判断主张是否已被覆盖，不产生本项目性能结论；
2. **作者报告数值**：只用于选择任务和估计上限，协议不一致时不得横向排名；
3. **本项目受控复现**：只有相同数据、planner、预算和多 seed 才能支持方法比较。

## 结果来源与口径

- **论文主结果**：LeWM 论文及官方项目页公布的规划成功率。
- **后续工作**：Fast-LeWM 项目页按相同四任务表格整理的结果，仅用于了解当前水平。
- **第三方复现**：公开 Hugging Face checkpoint 页面报告的 PushT 多种配置结果。
- **本项目结果**：Gemini 服务器上已验证的 TD-MPC2 CartPole 实验。

不同来源如果数据、观测、规划器、episode 数或成功条件不同，不直接进行横向排名。

## LeWM 论文四任务主结果

下表是最适合作为本项目目标参照的公开结果。数值为任务成功率（%）。

| 方法 | Two-Room | Reacher | PushT | Cube | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLDM | 97 | 78 | 78 | 65 | 79.5 |
| DINO-WM（pixels only） | 100 | 79 | 74 | 86 | 84.8 |
| LeWM | 87 | 86 | 96 | 74 | 85.8 |

该表揭示了几个重要事实：

1. **Two-Room 已接近饱和**。多数方法达到 97–100%，不适合作为唯一主任务。
2. **PushT 最能体现 LeWM 的优势**。LeWM 为 96%，PLDM 为 78%，纯像素
   DINO-WM 为 74%。
3. **Cube 更依赖视觉先验**。DINO-WM 为 86%，高于 LeWM 的 74%，说明当前
   LeWM 并非在所有任务上占优。
4. **Reacher 有一定区分度**。LeWM 为 86%，PLDM 和 DINO-WM 分别为 78% 和
   79%。

## PushT 的完整方法对照

LeWM 主图还报告了带本体状态的 DINO-WM 以及 goal-conditioned policy baseline：

| 方法 | PushT 成功率（%） |
| --- | ---: |
| LeWM | 96 |
| DINO-WM + proprioception | 92 |
| PLDM | 78 |
| GCBC | 75 |
| DINO-WM（pixels only） | 74 |
| GCIVL | 33 |
| GCIQL | 20 |
| Random | 2 |

论文的 PushT 汇总表给出 LeWM `96.0 ± 2.83`、PLDM `78.0 ± 5.0`、DINO-WM
`92.0 ± 1.63`。其中 92 对应带 proprioception 的 DINO-WM；不能与主图中 74 的
纯像素 DINO-WM 混为同一个设置。

PushT 是一个二维接触操作任务，动作为二维坐标，最长 200 steps；成功条件同时要求
位置误差小于 20 pixels、角度误差小于 `π/9`。不同公开数据版本和处理流程的 episode
数量可能不同，不能只凭数据集名称假定规模。本项目当前训练文件实际解析为 18,685
episodes；这里只记录规模元信息，不在 GitHub 保存数据文件。

## Cube、Two-Room 和 Reacher 补充结果

| 任务 | LeWM | PLDM | DINO-WM | GCBC | GCIQL | GCIVL | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-Room | 87 | 97 | 100 | 100 | 100 | 100 | 0 |
| Reacher | 86 | 78 | 79 | — | — | — | 10 |
| Cube | 74 | 65 | 86 | 84 | 64 | 56 | 48 |

Cube 的随机成功率已经达到 48%，因此仅看绝对成功率容易误判；报告时还应关注相对
随机策略的提升、任务初始状态分布和成功判定。

## LeWM 消融实验提供的设计信息

PushT 消融结果显示，稳定性和结构选择会显著改变结果：

| 设置 | PushT 成功率（%） |
| --- | ---: |
| Predictor small | 96.0 ± 2.83 |
| Predictor tiny | 80.67 ± 6.54 |
| Predictor base | 86.7 ± 3.06 |
| 不使用 decoder loss | 96.0 ± 2.83 |
| 使用 decoder loss | 86.0 ± 7.54 |
| ViT encoder | 96.0 ± 2.83 |
| ResNet-18 encoder | 94.0 ± 3.27 |
| Dropout 0.0 | 78.0 ± 6.54 |
| Dropout 0.1 | 96.0 ± 2.83 |
| Dropout 0.2 | 85.33 ± 5.74 |
| Dropout 0.5 | 66.67 ± 4.11 |

这说明“更大模型”或“增加重建损失”不一定更好。后续改进必须一次只改变一个因素，
并记录多 seed 方差。

规划器也会显著影响结果：

| Solver | LeWM | PLDM |
| --- | ---: | ---: |
| CEM | 96.0 ± 2.83 | 78.0 ± 5.0 |
| SGD | 26.0 ± 4.32 | 4.67 ± 0.06 |
| RMSProp | 67.33 ± 2.49 | 49.33 ± 8.26 |
| Adam | 84.0 ± 7.12 | 80.0 ± 3.27 |

因此比较 world model 时必须固定 solver、规划 horizon、候选数和优化迭代数，否则
测到的可能主要是 planner 差异。

## 第三方 PushT 复现

第三方公开 checkpoint 使用 3 个 seeds、每个 seed 50 个评测 episodes，报告：

| Backbone / frame stack | 3-seed mean ± std（%） |
| --- | ---: |
| ViT-S / 5 | 94.67 ± 2.31 |
| ViT-S / 10 | 70.67 ± 6.11 |
| ViT-S / 20 | 50.67 ± 3.06 |
| ViT-B / 5 | 88.00 ± 2.00 |
| ViT-B / 10 | 74.00 ± 4.00 |
| ViT-B / 20 | 42.00 ± 5.29 |

ViT-S、5-frame 的 94.67% 接近论文的 `96.0 ± 2.83`。同时，更多输入帧反而明显
降低成功率，这是值得在我们自己的复现中重点核查的数据处理和时序建模现象。

该结果来自第三方页面，不应写成官方复现结论；但它提供了可审计的 checkpoint、seed
和评测预算，可作为官方模型之外的交叉验证。

## 后续 Fast-LeWM 结果

后续 Fast-LeWM 工作在相同四任务表格中报告：

| 方法 | Two-Room | Reacher | PushT | Cube | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LeWM | 87 | 86 | 96 | 74 | 85.8 |
| Fast-LeWM | 98 | 88 | 96 | 80 | 90.5 |
| Fast-LeWM + SC | 98 | 90 | 98 | 82 | 92.0 |

它说明超过 LeWM 的公开目标已经提升到四任务平均 90.5%–92.0%。不过这是后续方法，
当前阶段应先作为上限参照，不应跳过原始 baseline 复现直接与它比较。

## 2026 年直接 LeWM/JEPA 对照版图

Fast-LeWM 已不是唯一需要考虑的后续工作。下列方法分别改变了监督、representation、
predictor 或 planner cost，不能只把作者报告的最好数字追加到同一个排行榜：

| 方法 | 主要新增信号 | 是否使用真实 reward/状态监督 | 应在哪个比较中出现 |
| --- | --- | --- | --- |
| Value-Guided JEPA | goal-conditioned value / quasi-distance | goal-reaching value | reward-labelled 或 goal-value 协议 |
| Reward-free bisimulation JEPA | transition-behavior equivalence | 无环境 reward；通常配冻结视觉 encoder | 视觉 OOD 与 control-invariance 专项 |
| RC-Aux | multi-horizon、budget reachability、temporal negatives | reward-free trajectories | 固定数据的 LeWM planning 对照 |
| Temporal-Distance JEPA | directed temporal cost、cross-trajectory negatives、rollout consistency | reward-free trajectories | 固定 planner 与 cost-form 分离比较 |
| Fast-LeWM | action-prefix multi-horizon prediction | reward-free trajectories | rollout error、规划速度和 success |
| PhyLatent | physical grounding、counterfactual separation、denoising | 使用物理状态 grounding | physical-representation 上限；不能冒充无监督公平对照 |
| PSG-JEPA | proprioception、joint-angle change grounding | 使用本体状态监督 | probe、policy 和真机专项 |

截至 2026-08-12，PhyLatent 作者报告 OGBench-Cube MPC success `70.0% -> 78.1%`、
TwoRooms `81.0% -> 98.0%`；Temporal-Distance JEPA 作者报告在锁定评测下 Two-Room
达到 100%，并在共享欧氏规划下使 OGB-Cube 相对 LeWM 提升 14.2 个百分点。这些数字的
baseline checkpoint、训练数据和监督并不自动等同于 Stable World Model 表格口径，只有
复现或逐项核对配置后才能进入正式结果表。

## 我们服务器上现有结果

Gemini 服务器已有 TD-MPC2 CartPole 结果，并已启动单 seed 的 LeWM PushT 从头训练。
LeWM 运行尚未完成，且启动于当前 package-only 规则确定之前，只能作为过渡工程实验：

| 实验 | 结果 | 定位 |
| --- | ---: | --- |
| TD-MPC2 CartPole sparse，seed 1，100k | best 4.67；final 0.00 | 探索失败案例 |
| TD-MPC2 CartPole dense，seed 1，100k | 864.55 | 工程 smoke baseline |
| Dense checkpoint 独立重载评测，3 episodes | 864.47 | 保存/加载链路验证 |
| LeWM PushT，seed 3072，10 epochs | 进行中 | 过渡训练，尚无最终评测结果 |

所以目前仍不能声称已经复现论文主任务。已有 CartPole 结果只证明基础训练、规划和
checkpoint 链路可用；LeWM 的实时状态和偏差见
[`baseline_lewm_pusht_training.md`](baseline_lewm_pusht_training.md)。

## 推荐的复现顺序

### 第一阶段：PushT checkpoint 评测

1. 下载官方 LeWM PushT checkpoint 和数据。
2. 先用 1 个 seed、10 episodes 跑通环境、观测、动作、规划和成功率统计。
3. 再按官方口径评测 3 seeds × 50 episodes，目标复现 `96.0 ± 2.83` 附近结果。
4. 保存逐 episode return、success、长度、规划耗时和显存峰值，而不只保存均值。

官方 Hugging Face PushT 数据仓库约 13.1 GB。当前约 6 GB vGPU 更适合先进行低并发
checkpoint 评测；从头训练 ViT world model 前需要另行测量显存和训练时间。

### 第二阶段：同协议 baseline 对照

按完全相同的数据 split、frame stack、CEM 预算和评测 seeds，依次评测：

1. PLDM；
2. DINO-WM（pixels only）；
3. DINO-WM + proprioception；
4. LeWM。

这一步的目标不是立刻超过论文，而是确认我们机器上的相对排序是否与公开结果一致。

### 第三阶段：从头训练和稳定性检查

至少运行 3 个训练 seeds，并检查：

- one-step prediction 和长时 open-loop rollout；
- representation collapse、NaN、梯度和 latent norm；
- checkpoint resume 是否改变后续结果；
- planner compute 是否完全一致；
- 成功率均值、标准差和最差 seed；
- 训练/推理时间、显存和规划吞吐。

### 第四阶段：加入第二个主任务

建议加入 Cube。PushT 更偏接触动力学和规划，Cube 更能暴露视觉表示不足。只有在这两类
任务上都优于 baseline，才更有资格声称新方法整体更好。

## 当前决定

- **工程 smoke baseline**：TD-MPC2 CartPole dense。
- **第一个正式论文 baseline**：LeWM PushT checkpoint evaluation。
- **第一组公平对照**：LeWM、PLDM、DINO-WM pixels-only、DINO-WM + proprioception。
- **第二个正式任务**：Cube。
- **主要评价**：多 seed 成功率、长时 rollout、稳定性、规划成本和 checkpoint resume。

## 公开来源

- Stable World Model：<https://github.com/galilai-group/stable-worldmodel>
- LeWM 论文：<https://arxiv.org/abs/2603.19312>
- LeWM 官方代码：<https://github.com/lucas-maes/le-wm>
- LeWM 官方项目页：<https://le-wm.github.io/>
- PushT 环境说明：<https://galilai-group.github.io/stable-worldmodel/envs/pusht/>
- 官方 LeWM PushT 数据与 checkpoint：<https://huggingface.co/datasets/quentinll/lewm-pusht>
- 第三方 PushT checkpoint 复现：<https://huggingface.co/MasonJK99/lewm-pusht>
- Fast-LeWM 项目页：<https://fast-lewm.github.io/>
- Value-Guided JEPA：<https://arxiv.org/abs/2601.00844>
- Reward-free bisimulation JEPA：<https://arxiv.org/abs/2602.18639>
- RC-Aux：<https://arxiv.org/abs/2605.07278>
- Temporal-Distance JEPA：<https://arxiv.org/abs/2607.25337>
- PhyLatent：<https://arxiv.org/abs/2608.05720>
- PSG-JEPA：<https://arxiv.org/abs/2608.06799>

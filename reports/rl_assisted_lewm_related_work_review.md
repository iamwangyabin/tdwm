# RL 辅助 LeWM：相关工作、比较矩阵与创新性审计

调研截止：2026-08-12<br>
文档状态：文献审计完成，实验复现尚未开始<br>
对应提案：[`rl_assisted_lewm_research_proposal.md`](rl_assisted_lewm_research_proposal.md)

## 1. 调研问题与结论

本次调研不再问宽泛的“RL 能不能帮助 LeWM”，而是逐项检查下面五个可能的论文主张：

1. RL/TD 信号能否让 latent world model 更适合控制；
2. reward-free TD 能否帮助 LeWM 学习长期动力学；
3. RL 能否防止 LeWM 表示坍塌；
4. RL 能否让 LeWM latent 更有物理意义；
5. 显式可滚动 LeWM 与 zero-shot successor model 的结合是否仍有未被覆盖的空间。

结论比原提案更保守：

> **“RL 帮助 LeWM”本身没有足够新颖性。** value-aware model learning、MuZero、
> TD-MPC、DeepMDP/DBC、SPR、TD-JEPA、RLDP、value-guided JEPA、reward-free
> bisimulation JEPA、PhyLatent 和 PSG-JEPA 已经分别覆盖了价值校准、控制充分表示、
> RL 与潜在预测联合训练、长期 successor prediction、防坍塌和物理 grounding。

目前仍可能成立、但必须通过公式级查重和实验验证的窄问题是：

> 在固定的无奖励离线视觉数据上，能否同时保留一个可对任意动作序列进行 MPC rollout
> 的局部 LeWM 动力学模型，以及一个面向任务/策略族的长期 successor model，并通过
> **model rollout 与真实数据的 successor response 一致性**校准前者；这种耦合是否比
> frozen LeWM + successor policy、TD-JEPA、RLDP、value-aware model loss 和近期
> plan-aware LeWM auxiliary 更好地支持未见 reward、长时规划与动力学变化？

这不是已经成立的新贡献，只是调研后剩下的候选空白。尤其需要正面排除两个替代解释：

- LeWM/RLDP 的局部 latent prediction 本身已经足以学习 zero-shot task features；
- successor/value head 只改进了 policy 或 planner，并没有改进可滚动 world model。

## 2. 调研方法与证据口径

### 2.1 检索范围

检索覆盖七条研究线：

1. decision-aware / value-aware / policy-aware model learning；
2. value-equivalent model 与面向规划的抽象模型；
3. bisimulation、行为相似度与 control-sufficient representation；
4. successor features、forward-backward representation 与 zero-shot RL；
5. self-predictive RL 与 latent dynamics auxiliary；
6. MuZero、TD-MPC、Dreamer 等 RL 训练的 latent world model；
7. DINO-WM、PLDM、LeWM 及 2026 年的直接 JEPA world-model 改进。

检索词除了方法名，还包括 `value-aware model learning`、`policy-aware model learning`、
`successor consistency`、`successor occupancy world model`、`reward-free JEPA planning`、
`latent dynamics zero-shot RL` 和 `physical grounding JEPA world model`。没有检索到一篇
与“可滚动视觉 JEPA + 独立多策略 successor model + model-vs-data successor consistency”
完全同构的工作，但这不构成新颖性的证明。

### 2.2 来源等级

- 优先使用论文正式会议页面、PMLR、NeurIPS/ICLR Proceedings、JMLR 和 arXiv 原文；
- 2026 年新工作多为 arXiv 预印本，结果必须标记为作者报告，不能当成独立复现；
- Stable World Model 网页上的表格只说明上游当前公开口径，不能替代本项目复现；
- 文献结果、第三方复现和本项目结果必须分表，不能混为一个排行榜。

### 2.3 比较维度

每项工作按以下维度比较，而不只比较最终成功率：

| 维度 | 需要回答的问题 |
| --- | --- |
| 训练监督 | 使用真实 reward、目标标注、本体状态，还是完全无奖励？ |
| 数据协议 | 在线交互、离线固定数据，还是离线后允许少量适配？ |
| 输入 | 低维状态、像素、冻结视觉特征，还是端到端像素？ |
| 学到的对象 | 一步 dynamics、任意动作序列 rollout、policy-conditioned successor、value 或 policy？ |
| RL 梯度路径 | 只进入 head/planner，还是进入 encoder 和 transition model？ |
| 测试方式 | MPC、直接 policy、reward fitting、fine-tuning，还是 goal-distance planning？ |
| 泛化类型 | 未见 goal、未见 reward、未见 policy、视觉 OOD 或动力学 OOD？ |
| 方法归因 | 提升来自 world model、表示、policy、planner、数据覆盖还是额外监督？ |

## 3. 研究版图

### 3.1 Decision-aware model learning：最直接的理论前身

这一研究线早已指出，最小化一般的 transition prediction loss 不一定等于最小化控制
误差。因此，“用价值函数衡量 world-model error”不是新想法。

| 工作 | 核心机制 | 与本项目的重合 | 关键差异或限制 |
| --- | --- | --- | --- |
| VAML, 2017 | 用 value-aware loss 代替纯概率模型损失 | 直接支持“模型误差应按决策后果衡量” | 原始设定不是端到端视觉 JEPA，也不以 reward-free 任务族为中心 |
| IterVAML, 2018 | 交替扩展 value/function 集合并学习模型 | 接近“函数族逐步校准模型” | 仍是任务价值驱动的模型学习 |
| PAML, 2020 | 根据 policy-gradient planner 如何使用模型定义 model loss | RL/planner 反向决定模型应准确的方向 | planner-specific，不是任意 reward 的 successor 表示 |
| Minimax Model Learning, 2021 | 从 off-policy policy evaluation 推导 transition-model loss | 与固定离线数据、分布偏移直接相关 | 重点是 OPE/OPO 与模型 misspecification |
| VaGraM, 2022 | 用 value gradient 重加权状态预测误差 | 与“控制相关方向优先”高度相似 | 使用给定任务 value；不保留 reward-free 通用性 |
| TOM, 2023 | 匹配真实环境与模型内当前策略的 transition occupancy | 与“model-vs-data occupancy consistency”非常接近 | 关注当前策略和在线 MBRL，不是多任务视觉 JEPA |
| Calibrated VAML, 2025 | 证明常用 value-aware/MuZero 式 loss 可能是不校准 surrogate | 直接否定“只要 value loss 降低，模型就更正确” | 给出理论警告，必须落实到本项目损失设计 |

这里对提案的影响最大：原先的 `L_BC` 若只是比较预测 latent 和真实 latent 经同一个 value
head 后的标量响应，很可能只是 VAML/MuZero-style surrogate 的一个实例，不能单独作为
方法贡献。至少需要说明：

- 使用的是何种 task/policy function class；
- 一致性对应 value、successor feature 还是 occupancy measure；
- 在何种条件下它比 latent MSE 提供新的约束；
- 如何避免 Calibrated VAML 指出的 surrogate calibration 问题；
- 与 TOM 的 transition-occupancy matching 在目标和数据分布上有何实质差异。

### 3.2 Value equivalence：模型不必还原全部世界

Value Equivalence Principle 把两个模型定义为：对给定 policy 和 function 集合产生相同
Bellman update。函数和策略集合越丰富，可接受的 model class 越接近真实环境。Proper
Value Equivalence 与 Value-Equivalent Sampling 又进一步讨论了正确的等价条件以及
模型容量、采样和规划之间的权衡。

这与本项目“让 LeWM 预测转移和真实转移在 Bellman functionals 下等价”的表述几乎是
同一理论语言。因此不能声称提出 value equivalence；能够尝试的新内容只能是：

- 将它实例化到从像素端到端训练的 reward-free LeWM；
- 使用多策略 successor response 而不是单任务 scalar value；
- 同时保留 prediction/SIGReg，研究 prediction fidelity 与 decision equivalence 的权衡；
- 用明确的固定 planner 实验证明变化发生在 rollable dynamics，而不是外挂 value head。

MuZero、Predictron、Value Prediction Network 等也早已学习不必重建观测、但对 planning
目标足够的抽象模型。后续分析还发现 MuZero 模型可能只在训练 policy/search 分布附近
准确，对未见 policy 泛化较弱。这提醒我们：task/policy family 的覆盖不是实现细节，
而是主方法能否支持 zero-shot 泛化的核心假设。

### 3.3 Control-sufficient representation：物理完整不等于控制充分

| 工作 | 表示原则 | 是否需要 reward | 与候选方案关系 |
| --- | --- | ---: | --- |
| DeepMDP, 2019 | reward prediction + latent transition distribution，理论连接 value preservation | 是 | 早期“latent dynamics + control-relevant representation”前身 |
| DBC, 2021 | latent 距离匹配 reward 与 transition 的 bisimulation distance | 是 | 已覆盖 decoder-free、抗干扰、任务相关不变性 |
| MICo, 2021 | 可采样的行为距离塑造 RL 表示 | 是 | 说明 value/behavior similarity 可直接监督 encoder |
| PSE, 2021 | 用 optimal-policy similarity 定义状态相似性 | 是/任务族 | 接近以 policy family 决定表示 |
| BS-MPC, 2025 | 在 MPC encoder 中加入 bisimulation metric loss | 是 | 与“RL 改善用于 MPC 的 encoder”直接重合 |
| Reward-free bisimulation JEPA, 2026 | 在 JEPA WM 上联合学习转移行为相似且抑制 slow features 的 encoder | 否 | 已占据“无奖励、control-relevant、JEPA planning”表述 |

2026 年的 *Learning Invariant Visual Representations for Planning with Joint-Embedding
Predictive World Models* 尤其关键。它在 JEPA/DINO-WM 上加入 bisimulation encoder，
让相似转移行为的状态接近，同时去除背景和 slow features，且不需要 reward predictor。
因此，下列表述已经站不住：

- 首次用无奖励控制结构改善 JEPA 表示；
- 首次让 JEPA 忽略任务无关视觉因素；
- 首次将 bisimulation/control equivalence 用于 JEPA planning。

本项目若继续，应把“物理有意义”拆成至少三个不同概念：

1. **物理可辨识性**：位置、角度、速度等可由 latent probe 解码；
2. **反事实动作敏感性**：不同动作的后果在 latent 中可区分；
3. **决策充分性**：会改变长期最优行为的状态差异被保留。

RL/successor signal 主要针对第三项，不能替代前两项的直接测量。

### 3.4 Zero-shot RL 与 successor representations

这一研究线与“无奖励 TD 帮助 LeWM”最接近，也是原比较中缺失最严重的部分。

| 工作 | 预训练产物 | 测试时适配 | 是否显式 rollable WM | 对本项目的含义 |
| --- | --- | --- | ---: | --- |
| Successor Features, 2017 | policy 的 discounted feature occupancy | 新 reward 权重 + GPI | 否 | 奠定 dynamics/reward factorization；不是新概念 |
| FB, 2021 | forward/backward successor-measure factorization与 policy family | 从 reward samples 得到任务向量，直接 policy | 否 | 已实现 reward-free TD 到任意后验 reward 的主张 |
| Does ZS-RL Exist?, 2023 | 系统比较 SF 与 FB | zero-shot | 否 | 发现 SF 对 state features 高度敏感，FB 依赖 replay coverage |
| ICVF, 2023 | 从 passive data 学 latent intentions/value factorization | downstream policy/value | 否 | 说明 action-free/off-policy 数据也能学多意图长期结构 |
| HILP, 2024 | 保留 temporal distance 的 Hilbert 表示与 foundation policies | goal/reward prompting | 否 | 已覆盖无奖励长期可达几何 |
| FRE, 2024 | reward function encoder + conditioned policy | 少量 reward-labelled samples | 否 | 给任意新 reward 的 functional encoding 是强基线 |
| Conservative ZS-RL, 2024 | 保守 FB/zero-shot policy | zero-shot | 否 | 低质量离线数据会导致 OOD action 高估，不能忽略 |
| TD-JEPA, 2025/2026 | state/task encoders、policy-conditioned multi-step predictor、policies | reward embedding 后直接 policy | 否 | 已覆盖 reward-free、offline、像素、multi-policy TD latent prediction |
| RLDP, 2026 | 正则化多步 latent dynamics features，再训练 BFM | reward fitting 后直接 policy | 训练期有 dynamics predictor，但测试不依赖 MPC | 证明简单 latent prediction + 防坍塌可匹配复杂 zero-shot 表示 |
| One-step FB, 2026 | 固定 behavior successor ratio 后做一步 policy improvement | zero-shot/fine-tune init | 否 | 揭示 joint FB 的循环依赖与收敛问题 |

#### TD-JEPA 与 LeWM 不是简单的“一个多步、一个一步”

TD-JEPA 的 predictor 近似给定 policy 的 successor features：它输入当前状态、动作和任务/
policy code，通过 TD bootstrap 预测该 policy 后续访问的长期 latent 总和。它不是一个能
接受任意未来动作序列并逐步产生 `z_{t+1:t+H}` 的控制模型。

LeWM 的 transition model 则是 policy-independent 的局部 action-conditioned model，可由
CEM/MPPI 在测试时组合任意候选动作序列。二者的抽象分别接近：

```math
\text{LeWM: } z_{t+1}=F(z_t,a_t),
\qquad
\text{TD-JEPA: } \Psi^{\pi_w}(z_t,a_t)
=\mathbb E_{\pi_w}\sum_{k\ge 0}\gamma^k\phi(z_{t+k+1}).
```

这说明两者具有互补性，但也说明“把 TD-JEPA 加到 LeWM”不够具体。真正需要检验的是：
显式局部模型与直接长期 occupancy model 同时存在时，能否通过交叉约束得到任一单模型
没有的能力。

#### RLDP 是最危险的反例

RLDP 直接研究“latent next-state prediction 是否足以为 zero-shot RL 学到 state features”。
作者报告：朴素 latent prediction 会出现 feature similarity 增加的轻度坍塌；加入简单
orthogonality regularization 后，可在多种 zero-shot RL 任务上匹配或超过更复杂的
successor-measure representation，并在低覆盖数据上更稳。

LeWM 的 prediction + SIGReg 与 RLDP 的研究命题高度接近。虽然 LeWM 面向像素 MPC、
RLDP 面向 behavioral foundation policy，方法细节不同，但它强制引入一个新的零假设：

> **冻结 LeWM 表示并在其上训练同一个 successor/policy stack，可能已经足够。**

如果 joint TD gradient 进入 LeWM 后没有超越这个 frozen baseline，就不能说 RL 改善了
LeWM；最多只能说 LeWM 是 zero-shot RL 的一个好 feature learner。

#### One-step FB 暴露联合训练风险

2026 年的 *Can We Really Learn One Representation to Optimize All Rewards?* 分析了 FB
中 representation、successor measure 和 improving policy 之间的循环依赖，并报告固定
behavior policy 的简化方案收敛更稳定。对应到本项目，第一版不应同时从随机任务方向
学习 encoder、successor、最优 policy 和 world model。更稳妥的顺序是：

1. 固定 LeWM 或 EMA encoder；
2. 先对 behavior policy 学 successor consistency；
3. 再引入 one-step policy improvement；
4. 最后才评估多策略 joint co-calibration。

### 3.5 Self-predictive RL：RL 与自监督潜在预测联合训练已有成熟先例

| 工作 | 机制 | 关键发现 | 对本项目的约束 |
| --- | --- | --- | --- |
| PBL, 2020 | 多步预测 bootstrapped future latents | 改善 multitask RL 表示 | “RL + latent prediction”不是新组合 |
| SPR, 2021 | EMA target + action-conditioned multi-step latent prediction | 提升像素 RL 样本效率 | 与 LeWM predictor 作为 RL auxiliary 直接相关 |
| EfficientZero, 2021 | MuZero targets + self-supervised latent consistency | 无人类数据 Atari 提升 | value/policy 与 self-prediction 可联合已有先例 |
| Understanding Self-Predictive RL, 2023 | 分析 predictor 速度和 semi-gradient | 优化动力学决定是否坍塌；理想化下学习 transition spectral structure | 防坍塌不能只归因于 TD 语义 |
| When Does Self-Prediction Help?, 2024 | 比较 reconstruction、自预测、TD 与 distractor | self-prediction 更适合作为 auxiliary，单独使用不总是最佳 | 需要强 frozen/auxiliary baseline |
| Action-Conditional SPR Framework, 2025 | 统一 fixed-policy 与 action-conditioned self-prediction | 两者对应不同低秩动力学与修改后的 value/Q | 理论区别必须基于具体 operator |
| OG-SPR, 2026 | latent self-prediction + observation grounding + adapters | 直接约束 shared representation 可能过强，adapter 更稳 | 建议 RL 梯度先走 adapter/residual，而非覆盖主 latent |

因此，“使用 RL 来帮助 SSL”只能是背景，不是贡献。值得研究的是 RL 信号经过哪里：

- 只训练 value/successor head；
- 通过小型 adapter 影响表示；
- 只校准 transition residual；
- 端到端更新 encoder 与 dynamics。

这四种设置必须分开，否则无法解释提升来自 representation、model 还是 policy。

### 3.6 RL 训练的 latent world models

| 工作 | 学习信号 | 是否可 rollout | 测试控制 | 与候选方案的边界 |
| --- | --- | ---: | --- | --- |
| Predictron/VPN | reward/value 与内部 planning targets | 抽象 rollout | value planning | 已覆盖“只预测决策所需量” |
| MuZero | reward、value、policy targets | 是 | MCTS | value-equivalent latent model 的代表；依赖任务 reward/self-play |
| Dreamer 系列 | reconstruction/reward/dynamics + imagined actor-critic | 是 | actor 或 imagination | RL 在 learned WM 内优化 policy，不等于 RL 校准 reward-free WM |
| TD-MPC | latent consistency、reward、Q、policy | 是 | MPC + terminal value | 最接近“TD 帮助可滚动 latent model”，但任务特定且有 reward |
| TD-MPC2 | 可扩展的多任务 task-oriented latent model | 是 | MPPI/CEM + actor/Q | 强系统 baseline，但不能与 reward-free arbitrary-goal 协议混排 |
| MR.Q | dynamics/reward prediction 学 value-friendly representation，不规划 | 训练有预测，测试无 MPC | model-free actor-critic | 说明 predictive representation 可能比 search 本身更关键 |

TD-MPC/TD-MPC2 已经充分说明 TD/value loss 可以训练用于 MPC 的 latent dynamics。候选
方案相对它的唯一重要区别是：不使用环境真实 reward，而是希望从 task/policy family
获得任务无关的长期约束。若实验只在单一 goal reward 上进行，这一区别立即消失。

### 3.7 直接 JEPA/LeWM 改进：2026 年竞争最密集的区域

| 工作 | 解决的问题 | 监督与主要机制 | 与候选方案的重合 |
| --- | --- | --- | --- |
| DINO-WM, 2025 | 稳定视觉 latent planning | 冻结 DINOv2 + action-conditioned predictor | 强视觉先验 baseline |
| PLDM, 2025 | 端到端 JEPA 稳定性 | prediction + VCReg + temporal + inverse dynamics | 多辅助项端到端 baseline |
| Value-guided JEPA, 2025/2026 | latent distance 不等于 goal value | 用 goal-reaching value 塑造 embedding distance | 直接占据 `LeWM + goal value loss` |
| Reward-free bisimulation JEPA, 2026 | slow features 和视觉 distractor | 无奖励 transition-behavior equivalence | 直接占据 reward-free control-aware JEPA |
| LeWM, 2026 | 从像素端到端稳定训练 | next-latent prediction + SIGReg | 已处理 global collapse，并报告 physical probes |
| Temporal Straightening, 2026 | latent path geometry 不利于规划 | 让时间轨迹更直 | 占据几何/时间结构改造 |
| RC-Aux, 2026 | predictive but not plannable | multi-horizon rollout + budget reachability + negatives | 占据 reachability 和 planning-aligned auxiliary |
| Sub-JEPA, 2026 | 高维 Gaussian regularization 代价/结构 | 子空间 Gaussian regularization | 更直接针对 non-collapse regularizer |
| SD-JEPA, 2026 | progression 与 content 混合 | 分解 latent subspaces | 占据结构化表示分解 |
| Fast-LeWM, 2026 | 自回归误差与规划速度 | action-prefix 多时域并行预测 | 占据多时域 rollable predictor |
| AdaJEPA, 2026 | test-time distribution shift 下 frozen world model 失效 | MPC 闭环中用新观测到的真实转移做 self-supervised test-time adaptation | 已占据“用少量新 transition 适配 LeWM dynamics”的路线 |
| Temporal-Distance JEPA, 2026 | 欧氏 latent cost 不反映时序进展 | directed temporal cost + negatives + rollout consistency | 占据 reward-free plan-aware distance |
| PhyLatent, 2026-08 | global non-collapse 不保证物理状态/动作后果 | physical grounding、future alignment、counterfactual separation、denoising | 直接占据“防物理坍塌、动力学相关表示” |
| PSG-JEPA, 2026-08 | 单 latent/latent pair 不可可靠识别机器人状态变化 | proprioception 与多时域关节变化 grounding | 直接占据“物理 grounding 改善规划和 policy” |

PhyLatent 在 2026-08-06 提出三类 failure mode：physical invariance collapse、physical
identifiability collapse 和 counterfactual dynamics collapse。作者报告其在 OGBench-Cube
上将 MPC success 从 70.0% 提升到 78.1%，TwoRooms 从 81.0% 提升到 98.0%。PSG-JEPA
在 2026-08-07 又从 proprioception 和多时域 joint-angle change 对表示进行物理 grounding，
并同时评测 probe、规划、policy learning 和真机。

这两篇工作使“RL 让 LeWM 真正有物理意义”不再是安全的主叙事。即使本项目最后使用
successor consistency，它更准确的表述也应是“改善 task-family decision sufficiency”，
而不是笼统地声称首次获得 physical semantics。

### 3.8 Reward-free exploration：另一条有效但不同协议的路径

Plan2Explore、ExORL、URLB、Proto-RL、METRA 等使用 intrinsic reward、技能发现或
探索策略改善数据覆盖。这的确是“RL 帮助 world model”的一种方式，但它改变了数据
分布和在线交互预算。

本项目主协议若固定离线数据，就不能把 exploration gain 混入方法效果。可以将其作为
独立后续问题：successor uncertainty 能否选择最有助于 LeWM 的新转移。届时必须使用
相同环境步数、相同采集成本和相同最终数据规模比较。

## 4. 最接近工作的逐项比较

### 4.1 候选方案的定义向量

为避免用模糊名称制造“差异”，先把候选方案写成八个属性：

| 属性 | 候选设定 |
| --- | --- |
| 数据 | 固定离线 transition dataset |
| 训练 reward | 不使用环境真实 reward |
| 观测 | 像素，端到端 encoder |
| 局部模型 | policy-independent、action-conditioned、可递归 rollout |
| 长期模型 | task/policy-conditioned successor response |
| 耦合 | 比较 model rollout 与 data transition 的 successor response |
| 测试 | 同时支持固定-cost MPC 与新 reward 的 policy/value inference |
| 主张 | RL signal 改善 rollable model 的 decision sufficiency，而不只是外挂 planner |

任何少于这些属性的表述都会被现有工作覆盖。

### 4.2 Closest-work matrix

符号：`是` 表示论文的核心能力；`部分` 表示有相近组件但不是主要协议；`否` 表示没有。

| 方法 | 无真实 reward | 固定离线 | 端到端像素 | 任意动作 rollout | 多策略 successor | RL 梯度进 dynamics | model-vs-data 决策一致性 | 最接近之处 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| VAML/VaGraM | 否 | 部分 | 否 | 是 | 否 | 是 | 是 | 用 value 衡量 model error |
| TOM | 否 | replay | 否 | 是 | 当前策略 | 是 | occupancy | model/data occupancy matching |
| Value Equivalence | 可抽象 | 可抽象 | 可抽象 | 是 | 可定义 | 是 | Bellman update | 理论语言几乎相同 |
| DeepMDP/DBC/BS-MPC | 否 | 部分 | 是 | 部分/是 | 否 | 是 | bisimulation | control-sufficient latent |
| TD-MPC2 | 否 | 在线/replay | 是 | 是 | 多任务但有 reward | 是 | TD/value | TD 训练 MPC latent model |
| FB/One-step FB | 是 | 是 | 部分 | 否 | 是 | 无显式 F | successor ratio | reward-free zero-shot policy |
| TD-JEPA | 是 | 是 | 是 | 否 | 是 | predictor 即 successor | TD successor | 多策略长期 latent prediction |
| RLDP | 是 | 是 | 部分 | 训练期是 | 后续 BFM | representation stage 分离 | 否 | latent dynamics features 足以做 ZS-RL |
| Value-guided JEPA | 否/goal cost | 是 | 是 | 是 | goal family | 是 | value-distance | value 塑造 JEPA latent |
| Reward-free bisim JEPA | 是 | 是 | 冻结 encoder 为主 | 是 | 否 | 是 | transition equivalence | 无奖励 control-aware JEPA |
| RC-Aux/TD-JEPA-distance | 是 | 是 | 是 | 是 | goal/reachability | 是 | rollout/temporal | plan-aware LeWM auxiliary |
| PhyLatent/PSG-JEPA | 部分无 reward；使用 physical labels | 是 | 是 | 是 | 否 | 是 | physical/counterfactual | 物理结构与动作后果 grounding |
| 候选方案 | 是 | 是 | 是 | 是 | 是 | 是 | successor response | 两种时间尺度模型的显式交叉校准 |

这个矩阵显示，候选方案没有一个单独组件是新的。可能的新颖性只存在于最后一列所述的
**组合关系和验证问题**，而不是在 TD、successor、JEPA、MPC 或 anti-collapse 任一术语。

## 5. 哪些主张已经不能写

| 不应再写的主张 | 直接阻断它的工作 | 可改写成的可检验问题 |
| --- | --- | --- |
| RL 首次帮助 latent WM 学控制表示 | DeepMDP、DBC、MuZero、TD-MPC、BS-MPC | reward-free successor signal 是否改善 LeWM 的 rollable dynamics |
| TD loss 首次改善 JEPA | TD-JEPA、value-guided JEPA | multi-policy successor consistency 是否与局部 action rollout 互补 |
| RL 防止 LeWM 特征坍塌 | SPR 理论、TD-JEPA non-collapse、RLDP、LeWM/SIGReg、Sub-JEPA | TD gradient 是否在 SIGReg 已稳定时保留更多 decision-relevant rank |
| 首次让 LeWM 表示有物理意义 | LeWM probes、PhyLatent、PSG-JEPA | successor signal 是否改善长期 decision sufficiency，且不损害 physical probes |
| 首次学习 reachability/temporal distance | HILP、VIP、RC-Aux、Temporal-Distance JEPA | 超出 goal-distance 的多 reward/policy consistency 是否有额外收益 |
| 首次用 Bellman functional 校准模型 | VAML、Value Equivalence、Calibrated VAML | 在 reward-free visual JEPA 上的 calibrated successor-response objective 是否成立 |
| 首次结合 model-based 与 successor features | SF 理论、Lehnert & Littman、hybrid MB-SF | 显式 co-training 是否优于并行但独立的 local model 与 successor policy |
| zero-shot 适配任意 reward | SF、FB、HILP、FRE、TD-JEPA、RLDP | 在明示 reward-basis span 和 inference budget 下是否改善 |

## 6. 调研后仍可能存在的研究空白

### 6.1 Local model 与 successor model 的互补，而不是简单拼接

显式 one-step model 能对任意 action sequence 组合和重规划；successor model 将特定
policy 的长时 occupancy 压缩为一次前向预测。前者计算更贵但对 policy 改变灵活，后者
计算快但对 transition/policy 变化脆弱。

候选科学问题可以写为：

> policy-independent local dynamics 与 policy-conditioned long-horizon dynamics 是否
> 能互相校准，使有限容量视觉表示同时保留短时反事实动作后果和长期任务相关结构？

### 6.2 Model–successor consistency，而不是普通 value loss

一个更清晰的候选目标是先定义在真实数据上学得并冻结的 successor operator
`S^pi_data`，再比较模型 rollout 诱导的 discounted feature occupancy：

```math
\hat z_{t+k+1}=F_\phi(\hat z_{t+k},a_{t+k}),
\qquad
\hat\Psi_H^{\pi,w}(z_t,a_t)
=\sum_{k=0}^{H-1}\gamma^k u_w(\hat z_{t+k+1})
+\gamma^H\bar\Psi^{\pi,w}(\hat z_{t+H},a_{t+H}),
```

```math
\mathcal L_{\mathrm{MSC}}
=\mathbb E_{w,\pi,H}
\left\|
\hat\Psi_H^{\pi,w}(z_t,a_t)
-\operatorname{sg}\Psi_{\mathrm{data}}^{\pi,w}(z_t,a_t)
\right\|^2.
```

这个目标与原来的单步 `G(hat z_{t+1}) - G(z_{t+1})` 有两个差异：

- 它显式检查可滚动模型在多步下诱导的 occupancy response；
- target 来自真实 transition TD estimation，而不是对两个 latent 调用同一个可漂移 head。

即便如此，它仍与 VAML、Value Equivalence 和 TOM 高度相关。论文贡献必须依赖 reward-free
视觉 JEPA、多任务/策略族、显式 local-vs-successor operator coupling 和严格归因实验，
不能只依赖这个公式的命名。

### 6.3 Reward revaluation 与 transition revaluation

Successor features 擅长 reward 改变，但固定 policy successor 对 transition 改变适应较慢；
显式 world model 理论上能在获得新 dynamics evidence 后重新规划。这个差异提供了比
普通 IID goal success 更有判别力的实验：

- reward revaluation：动力学不变，只改变下游 reward；
- policy revaluation：reward 不变，改变允许的 policy/action constraint；
- transition revaluation：视觉不变，改变质量、摩擦、障碍或动作响应；
- joint revaluation：同时改变 reward 与 dynamics。

需要注意：如果测试时完全不给新动力学数据，LeWM 也不会自动知道环境变了。因此必须
明示 adaptation budget，例如提供相同的少量无 reward transitions，并比较：

- frozen successor policy；
- successor model 少量 TD 更新；
- local world model 少量 dynamics 更新后重新规划；
- local + successor co-calibration。

AdaJEPA 已经在 MPC 重规划闭环中用新观测到的 transition 自监督更新 world model。
因此，transition revaluation 只能作为检验 local model 与 successor model 分工的诊断，
不能单独作为本项目的创新点；若将其写入主要实验，还必须加入 AdaJEPA 式适配对照，
共享 transition 数量、梯度步数、规划调用和总算力预算。

### 6.4 Gradient routing 可能比新 loss 更重要

OG-SPR 的结果提示，直接用 auxiliary self-prediction 约束 shared representation 可能
过强。候选方法应优先测试：

1. frozen LeWM + detached successor；
2. successor gradient 只进入 adapter；
3. successor gradient 只进入 dynamics residual；
4. successor gradient 进入 encoder；
5. 端到端全部共享。

如果 adapter 版本最好，论文的贡献应诚实定位为 stable coupling，而不是全局表示学习。

## 7. 必需比较，而不是可选 related work

### 7.1 协议 A：Reward-free arbitrary-goal planning

同一数据、同一 CEM/MPPI、同一 horizon 和同一 goal cost 下的 P0 baseline：

| 优先级 | 方法 | 原因 | 实现状态口径 |
| --- | --- | --- | --- |
| P0 | LeWM | 直接母体 | 必须通过 `stable_worldmodel==0.1.1` 公共 API 复现 |
| P0 | LeWM + detached successor/value head | 检查收益是否只是 planner augmentation | 本项目轻量 head，不修改上游 |
| P0 | LeWM + adapter successor gradient | 检查最小梯度路径 | proposed-method 消融 |
| P0 | RC-Aux 或 Temporal-Distance JEPA | 最接近 plan-aware LeWM auxiliary | 无统一公开接入时标注文献对照 |
| P0 | PhyLatent | 最接近 physical/dynamics-relevant LeWM 改进 | 需核对公开代码与数据协议 |
| P1 | PLDM | 端到端多辅助 JEPA | 上游 baseline |
| P1 | DINO-WM pixels-only | 冻结强视觉表示 | 上游 baseline |
| P1 | reward-free bisimulation JEPA | control-sufficient JEPA | 视觉 OOD 专项 |
| P1 | AdaJEPA | 测试时自监督 dynamics adaptation | 仅在 transition revaluation 协议中必需 |

主结论必须来自同一个 planner cost；`+ successor terminal value` 只能作为系统上限单列。

### 7.2 协议 B：Zero-shot reward transfer

P0 baseline：

| 方法 | 必需原因 |
| --- | --- |
| frozen LeWM + RLDP-style representation protocol + same successor stack | 最强零假设：LeWM latent 本身是否足够 |
| RLDP | 直接检验 latent dynamics prediction 作为 BFM feature learner |
| TD-JEPA | reward-free、offline、像素、多策略 TD latent prediction 的最近工作 |
| FB | 经典 successor-measure zero-shot baseline |
| One-step FB | 更稳定、打破 circular dependency 的 2026 对照 |
| HILP | temporal-distance representation 与 foundation policy |
| FRE | 任意 reward 的 functional encoding，对少量 reward samples 公平 |

所有方法必须共享：reward inference samples、是否允许 planning、policy 参数量、离线数据、
行为覆盖、测试 reward 集合与适配计算预算。直接 policy 和 MPC 的延迟也要分别报告。

### 7.3 协议 C：Reward-labelled control

TD-MPC2、MR.Q、BS-MPC、VaGraM 等属于真实 reward 协议。它们回答“任务 reward 能否
帮助 model/representation”，但不能直接证明 reward-free 泛化。若加入此协议，必须为
所有方法提供相同 reward labels 和环境 steps，并单独排名。

### 7.4 协议 D：Data acquisition

Plan2Explore、ExORL、METRA 等只在允许额外交互时比较。不能用更多、更优数据击败固定
数据 baseline 后，把收益归因于 world-model objective。

## 8. 最小归因实验矩阵

| 编号 | Representation | Local dynamics | Successor/policy | 交叉校准 | 测试控制 | 能回答的问题 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | LeWM | LeWM | 无 | 无 | 原始 MPC | 基线 |
| E1 | frozen LeWM | frozen LeWM | 训练 | 无 | 原始 MPC | head 是否不影响 WM |
| E2 | frozen LeWM | frozen LeWM | 训练 | 无 | successor policy/value | planner/policy 上限 |
| E3 | adapter 更新 | frozen LeWM | 训练 | 单步 | 原始 MPC | 表示 adapter 是否足够 |
| E4 | frozen encoder | dynamics residual | 训练 | 多步 MSC | 原始 MPC | RL 是否直接改善 rollable F |
| E5 | encoder + F | 联合 | 训练 | 多步 MSC | 原始 MPC | 完整 co-calibration 是否必要 |
| E6 | encoder + F | 联合 | 训练 | 多步 MSC | successor bootstrap MPC | 完整系统上限 |
| E7 | TD-JEPA | 无 local F | 联合 | TD successor | direct policy | 显式 local model 是否必要 |
| E8 | RLDP | 训练期 F | same successor stack | 无 | direct policy | dynamics features 是否已足够 |
| E9 | LeWM | LeWM | 参数量匹配随机 auxiliary | 无 | 原始 MPC | 额外参数/正则解释 |

最关键的成功条件不是 E6 的最终成功率，而是：E4/E5 在**不使用新 successor planner**时，
仍以同一 MPC cost 稳定优于 E0，并且超过 E1/E3/E8 能解释的收益。

## 9. 指标必须对应文献争议

### 9.1 防止“只测 success rate”

- one-step latent prediction；
- 5/10/20-step open-loop latent error；
- model rollout 与 data transition 的 held-out successor-response error；
- held-out policy 与 held-out reward 上的 error；
- fixed planner / fixed cost success；
- successor-augmented planner success；
- direct zero-shot policy success；
- 参数、训练 FLOPs、规划延迟、适配延迟和峰值显存。

### 9.2 防止“把 non-collapse 当成物理意义”

- covariance spectrum、effective rank 和 pairwise cosine similarity；
- 位置、速度、角度、本体状态的 linear/nonlinear probes；
- counterfactual action separation；
- static visual invariance；
- temporal/reachability ordering；
- policy/value linearity；
- 相同视觉但不同隐藏物理参数下的 state aliasing。

### 9.3 防止“数据覆盖决定一切”

- expert、mixed、random/low-quality replay 分开；
- action support 与 learned policy OOD rate；
- trajectory stitching 子集；
- 覆盖量匹配和数据规模曲线；
- conservative regularization 开/关。

## 10. 对原研究方案的修正

经过这次比较，原提案应做五项实质修改：

1. **弃用 BC-LeWM 名称。** BC 在 RL 中通常表示 behavior cloning，而且“Bellman
   Calibrated”与 VAML/Calibrated VAML/Value Equivalence 太接近，容易夸大差异。
2. **不再主打 anti-collapse 或 physical meaning。** 这两项已有 LeWM、RLDP、
   PhyLatent、PSG-JEPA 和 self-predictive RL 理论的直接覆盖。
3. **把单步 scalar Bellman loss 降级为 baseline。** 主候选改为多步 model–successor
   consistency，并明确 target 来自真实 transition TD estimator。
4. **将 frozen LeWM + RLDP/FB stack 提升为 P0 baseline。** 这是判断 joint gradient
   是否必要的最强对照。
5. **先固定 behavior policy，再逐步增加 policy improvement。** 避免 FB/联合 TD 的循环
   依赖和离线 OOD action 让模型与 critic 相互强化。

论文叙事也应从：

> 使用 RL 让 LeWM 学到更物理、更不坍塌的世界模型。

改成：

> We study whether a policy-independent, rollable JEPA dynamics model and a
> policy-conditioned successor model provide complementary abstractions of the same
> reward-free offline dynamics, and whether enforcing model–successor consistency improves
> long-horizon decision sufficiency beyond frozen predictive representations or planner-only
> augmentation.

中文：

> 我们研究同一份无奖励离线动力学的两种抽象是否互补：可组合任意动作序列的局部 JEPA
> 模型，以及压缩特定策略长期占用的 successor model；并检验二者的一致性是否真正改善
> 长时决策充分性，而不是只训练出更强的 policy 或 planner。

## 11. Go / No-Go 决策门

### Go：值得继续做完整方法

同时满足：

1. frozen LeWM 上的 successor estimator 在 held-out reward/policy 上可稳定学习；
2. local model rollout 与 data successor response 存在 latent MSE 没捕捉到的系统误差；
3. 多步 MSC 能降低该误差，且不是通过 task basis collapse 实现；
4. 同一 planner/cost 下 E4 或 E5 超过 E0、E1、E3 和参数量匹配对照；
5. 增益至少在未见 reward、deceptive geometry 或 dynamics revaluation 中一项稳定存在；
6. 不明显破坏 LeWM prediction、物理 probes 和视觉 OOD 鲁棒性。

### No-Go：应停止或改变论文主张

出现任一项：

- frozen LeWM + successor 已取得全部收益；
- RLDP/TD-JEPA 在相同协议下占优且 local rollout 没提供新能力；
- 只有使用 successor terminal value 时 success 提升；
- MSC 降低但 fixed-cost MPC、rollout 和 probes 不改善；
- task/policy basis 在离线数据上持续坍塌或被 OOD action 污染；
- 方法只在训练 reward 生效；
- 物理 grounding baseline 已解释所有收益。

此时更诚实的论文方向可能是：

- “LeWM is a strong representation learner for zero-shot RL”；
- “When does successor supervision harm or help rollable JEPA models?”；
- “A negative result on co-training local and successor dynamics from offline pixels”。

## 12. 当前推荐

这次调研后的推荐不是立刻实现完整 joint 方法，而是按下面顺序做三个低成本判别实验：

1. **LeWM vs RLDP-style feature learner**：相同数据和 latent 维度，统一 successor stack；
2. **frozen vs adapter vs dynamics-only gradient**：明确 RL 信号必须进入哪里；
3. **single-step value surrogate vs multi-step MSC**：检查真正有用的是 value shaping，
   还是 local model 与 long-horizon successor 的交叉一致性。

只有第三项在固定 planner/cost 下稳定成立，才有理由把“RL 帮助 LeWM”发展成一个新方法。

## 参考资料

### Decision-aware 与 value-equivalent model learning

- Value-Aware Loss Function for Model-based RL：<https://proceedings.mlr.press/v54/farahmand17a.html>
- Iterative Value-Aware Model Learning：<https://papers.nips.cc/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html>
- Policy-Aware Model Learning：<https://arxiv.org/abs/2003.00030>
- Minimax Model Learning：<https://proceedings.mlr.press/v130/voloshin21a.html>
- VaGraM：<https://arxiv.org/abs/2204.01464>
- Transition Occupancy Matching：<https://proceedings.mlr.press/v211/ma23a.html>
- Value Equivalence Principle：<https://arxiv.org/abs/2011.03506>
- Proper Value Equivalence：<https://arxiv.org/abs/2106.10316>
- Value-Equivalent Sampling：<https://arxiv.org/abs/2206.02072>
- Calibrated Value-Aware Model Learning：<https://proceedings.mlr.press/v267/voelcker25a.html>
- Predictron：<https://proceedings.mlr.press/v70/silver17a.html>
- Value Prediction Network：<https://arxiv.org/abs/1707.03497>
- MuZero：<https://arxiv.org/abs/1911.08265>
- What Model Does MuZero Learn?：<https://arxiv.org/abs/2306.00840>

### Control-sufficient representation

- DeepMDP：<https://proceedings.mlr.press/v97/gelada19a.html>
- DBC：<https://arxiv.org/abs/2006.10742>
- MICo：<https://papers.nips.cc/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html>
- Policy Similarity Embeddings：<https://arxiv.org/abs/2101.05265>
- BS-MPC：<https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea0206fdf3afc2ff0578a230816a9e15-Abstract-Conference.html>
- Reward-free bisimulation JEPA：<https://arxiv.org/abs/2602.18639>

### Successor features 与 zero-shot RL

- Successor Features for Transfer：<https://arxiv.org/abs/1606.05312>
- Successor Features Combine Model-Free and Model-Based RL：<https://www.jmlr.org/papers/v21/19-060.html>
- Forward-Backward Representation：<https://arxiv.org/abs/2103.07945>
- Does Zero-Shot RL Exist?：<https://arxiv.org/abs/2209.14935>
- ICVF：<https://proceedings.mlr.press/v202/ghosh23a.html>
- HILP：<https://arxiv.org/abs/2402.15567>
- FRE：<https://proceedings.mlr.press/v235/frans24a.html>
- Conservative Zero-Shot RL：<https://papers.nips.cc/paper_files/paper/2024/hash/1e38b2a0b77541b14a3315c99697b835-Abstract-Conference.html>
- TD-JEPA：<https://arxiv.org/abs/2510.00739>
- RLDP：<https://arxiv.org/abs/2603.15857>
- Can We Really Learn One Representation to Optimize All Rewards?：<https://arxiv.org/abs/2602.11399>
- Zero-Shot Adaptation to Unseen Dynamics：<https://arxiv.org/abs/2505.13150>

### Self-predictive RL 与 latent RL world model

- PBL：<https://proceedings.mlr.press/v119/guo20g.html>
- SPR：<https://arxiv.org/abs/2007.05929>
- EfficientZero：<https://papers.neurips.cc/paper_files/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html>
- Understanding Self-Predictive Learning for RL：<https://proceedings.mlr.press/v202/tang23d.html>
- When Does Self-Prediction Help?：<https://arxiv.org/abs/2406.17718>
- Action-Conditional Self-Predictive RL：<https://proceedings.mlr.press/v258/khetarpal25a.html>
- OG-SPR：<https://arxiv.org/abs/2608.05989>
- TD-MPC：<https://arxiv.org/abs/2203.04955>
- TD-MPC2：<https://arxiv.org/abs/2310.16828>
- DreamerV3：<https://arxiv.org/abs/2301.04104>
- MR.Q：<https://arxiv.org/abs/2501.16142>

### JEPA、LeWM 与规划

- DINO-WM：<https://arxiv.org/abs/2411.04983>
- PLDM：<https://arxiv.org/abs/2502.14819>
- LeWM：<https://arxiv.org/abs/2603.19312>
- Value-Guided JEPA：<https://arxiv.org/abs/2601.00844>
- Temporal Straightening：<https://arxiv.org/abs/2603.12231>
- RC-Aux：<https://arxiv.org/abs/2605.07278>
- Sub-JEPA：<https://arxiv.org/abs/2605.09241>
- SD-JEPA：<https://arxiv.org/abs/2605.31111>
- Fast-LeWM：<https://arxiv.org/abs/2606.26217>
- AdaJEPA：<https://arxiv.org/abs/2606.32026>
- Temporal-Distance JEPA：<https://arxiv.org/abs/2607.25337>
- PhyLatent：<https://arxiv.org/abs/2608.05720>
- PSG-JEPA：<https://arxiv.org/abs/2608.06799>
- Plan2Vec：<https://proceedings.mlr.press/v120/yang20b.html>
- VIP：<https://arxiv.org/abs/2210.00030>
- Stable World Model baselines：<https://galilai-group.github.io/stable-worldmodel/baselines/>

### Reward-free exploration

- Plan2Explore：<https://arxiv.org/abs/2005.05960>
- URLB：<https://arxiv.org/abs/2110.15191>
- ExORL：<https://arxiv.org/abs/2201.13425>
- METRA：<https://arxiv.org/abs/2310.08887>

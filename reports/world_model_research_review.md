# World Model 领域调研与 RL 辅助 LeWM 研究方案

调研截止：2026-08-12<br>
文档状态：领域调研和创新性审计完成，方法尚未确定，实验尚未开始<br>
项目平台：`stable-worldmodel[all]==0.1.1`

## 1. 执行摘要

本项目最初的问题是：能否用 reinforcement learning（RL）帮助 LeWorldModel（LeWM），
得到更适合控制、长期预测和泛化的世界模型？经过从 Dyna、经典 model-based RL、latent
state-space model、decision-aware model learning、successor representation、JEPA、
interactive video model、World Action Model、机器人与自动驾驶，到 2026 年最新 LeWM
后续工作和 world-model evaluation 的系统检索，结论需要比原始设想保守得多。

**有研究意义，但宽泛命题基本都已被覆盖：**

- “用 RL 或 value/TD 信号训练 world model”已有 MuZero、TD-MPC/TD-MPC2、Dreamer、
  VAML/VaGraM/TOM、RLVR-World 等直接前身；
- “用自监督 latent prediction 防止坍塌或支撑控制”已有 SPR、Self-Predictive RL、
  DreamerPro、MuDreamer、R2-Dreamer、TD-JEPA 和 RLDP；
- “让 LeWM 更适合规划”已有 Value-guided JEPA、Temporal Straightening、RC-aux、
  Temporal-Distance JEPA、Fast-LeWM 和 Hierarchical Planning；
- “让 LeWM 更物理、更不坍塌”已有 LeWM/SIGReg、Temporally Centered SIGReg、
  PhyLatent、PSG-JEPA 和 Metric Non-Collapse；
- “联合一步模型和长期 successor model”与 successor representation、`gamma`-models、
  TD-Flow、Universal Horizon Models 和 Jumpy World Models 高度重合；
- “联合预测未来和动作，让 world model 直接帮助 policy”已有 Video Prediction Policy、
  DreamGen、DreamZero、LaWAM、VLA-MBPO 和 latent-action world models；
- “视频逼真或 latent 满秩就说明理解物理”证据不足。WorldModelBench、WorldBench、
  WorldGym 和 MMBench2 共同暴露了物理违背、动作不服从、OOD policy value 偏乐观和
  visually fluent hallucination。

因此，**当前不能声称提出了新方法**。最值得保留的是一个待证伪的问题，而不是结论：

> 在固定的 reward-free offline visual trajectories 上，policy-independent、可接受任意
> action sequence 的 LeWM 局部模型，与 policy-conditioned 的长期 occupancy model 是否
> 提供可测量的互补性？若通过同一套 data-derived long-horizon targets 联合校准，能否在
> 不依赖新 reward head 或更强 planner 的情况下改善 model fidelity、规划与 revaluation？

这个问题仍然拥挤。尤其是 Jumpy World Models 已经学习 policy-conditioned、多时间尺度
occupancy，并引入跨时间尺度 consistency 用于组合规划；TD-Flow/UHM 已直接学习长期未来
分布；LaWAM 又把 latent future prediction 与动作生成连成了低延迟 policy。剩余空间只能
来自**端到端视觉 JEPA、primitive-action 任意序列可滚动性、与真实数据长期算子的双向
一致性，以及严格的 world-model 归因评测**这几个条件的同时成立。若公式级比较或 P0 实验
不能证明额外价值，应停止此方向，而不是换一个名字继续包装。

### 1.1 调研口径与完整性边界

- 资料以论文原文、正式 proceedings、作者项目页和 Stable World Model 官方文档为主；
- 从 2025-2026 年综述的 taxonomy 反查历史主线，再沿 LeWM、Dreamer、successor model、
  video world model、WAM 和 evaluation 做前向近邻检索；
- 覆盖 architecture、state/action/time representation、training signal、data regime、test-time
  use、failure mode 和 evaluation evidence，不把“列到论文名”当作已经完成比较；
- 当前共收录 140 个 primary/official references。2026 年工作大量仍是 preprint，文中将其
  用于 novelty audit，不把作者报告的结果当作本项目已复现事实。截止日期后的新论文仍需
  在实验开始前再做一次公式级检索。

## 2. 什么是 World Model

“World model”目前没有全领域一致定义。本文采用与 embodied decision-making 相容的工作
定义：它是学习环境结构与动力学、并能被 agent 查询或消费的内部 simulator。在控制语境中，
它不是“看起来合理的视频生成器”的同义词。它至少要保存某种可供决策使用的环境状态，并
预测动作如何改变未来。一个通用的部分可观测形式是：

```math
b_t = E(o_{\le t}, a_{<t}), \qquad
\hat b_{t+1} \sim F(b_t,a_t),
```

其中 `b_t` 可以是显式状态、belief、连续或离散 latent、objects/scene graph，甚至程序。
按用途，模型还可能预测 observation、reward、termination、constraint、uncertainty、value、
successor occupancy 或可达性。模型的价值不在于组件名称，而在于它能否支持下列一种或
多种操作：

1. **Online planning**：CEM、MPPI、MCTS、gradient planning 或 graph search；
2. **Policy learning in imagination**：用 imagined rollouts 训练 actor/critic；
3. **Synthetic data**：为 offline/online RL 生成短 rollout；
4. **Zero-shot task solving**：测试时改变 goal/reward，再规划或快速得到 policy；
5. **Exploration and data acquisition**：利用 novelty 或 uncertainty 选择真实交互；
6. **Simulation and evaluation**：作为 agent 的可控 learned environment。

这一区分非常重要：MuZero 的模型只需为规划保持 reward、value 和 policy 所需信息；
Dreamer 需要支撑 imagined policy learning；LeWM 需要在 latent 中滚动任意候选动作并由
goal cost 排序；successor model 则缓存某个 policy 下的长期 occupancy。它们都可称为
world model，但接口、监督和可适配性不同，不能只按“是否预测未来”混为一类。

从理论上也不能把“world model”限制为一个显式命名的 neural network。General Agents
Need World Models 证明，多步 goal-directed generalization 会在 policy 中隐含可提取的预测
模型。工程上真正需要区分的是模型是否**显式可查询**、能否接受 counterfactual actions、
预测什么对象，以及决策模块怎样消费预测。

| 设计轴 | 主要选项 | 直接后果 |
| --- | --- | --- |
| 状态 | fully observed state / history / belief | 决定是否能处理 partial observability |
| 输出 | state / latent / reward-value / occupancy / pixels-video | 决定 fidelity 与 task sufficiency |
| 空间 | global vector / feature tokens / spatial grid / objects / 3D-4D rendering | 决定几何、交互和计算成本 |
| 时间 | sequential rollout / multi-step / global difference / arbitrary horizon | 决定误差累积、组合性和推理成本 |
| 条件 | arbitrary actions / policy / skill / latent action | 决定 counterfactual planning 与复用范围 |
| 随机性 | deterministic / stochastic / ensemble / generative | 决定多模态未来与 uncertainty 表达 |
| 用法 | MPC/search / imagination RL / synthetic data / direct policy | 决定公平 baseline 和评价指标 |
| 数据 | online / offline / video-only / mixed embodiment | 决定 coverage、action labels 和可验证性 |

为避免“彻底”变成无边界罗列，本文覆盖的是**学习得到、面向 embodied decision/control 的
环境模型**，包括 compact latent MBRL、interactive video simulator、机器人与驾驶 world
model。纯天气/PDE 预测、无动作的普通视频生成、只讨论语言模型是否隐含世界知识的工作不
作为直接 baseline；它们只有在提供可查询 action/counterfactual interface 时才进入比较。
同样，global difference prediction、3D/4D rendering 和驾驶模型属于领域版图，但与
Stable World Model 上的 compact visual MPC 不是可直接混排的实验对象。

### 2.1 RL 到底可以怎样帮助 World Model

“RL 帮助 LeWM”至少有六种含义，必须先指定梯度、数据和测试接口，否则无法判断是否新：

| 入口 | RL 或 policy 提供的信号 | 改变的对象 | 直接前身 | 对本项目的适用性 |
| --- | --- | --- | --- | --- |
| Active data collection | novelty、uncertainty、failure | 真实数据 coverage | Dyna、Plan2Explore、MMBench2 | 固定数据协议外，单独实验 |
| Control-aware representation | reward、value、action、TD target | encoder/latent geometry | MuZero、TD-MPC、MuDreamer、TD-JEPA | 会改变 reward-free 定义 |
| Decision-aware model loss | value gradient、occupancy、planner usage | transition model | VAML、VaGraM、PAML、TOM | 可作强 baseline |
| Imagination policy learning | model rollout 上的 return/critic | actor、critic，间接改变模型访问分布 | Dreamer、TD-MPC2、VLA-MBPO | 属于完整 MBRL，不与纯 MPC 混排 |
| Temporally abstract prediction | TD recursion、policy occupancy | successor/horizon operator | `gamma`-model、TD-Flow、Jumpy WM | 与候选最接近 |
| Generator post-training/adaptation | action fidelity、verifiable reward、new transitions | video/latent generator 或 adapter | RLVR-World、RLIR、AdaJEPA、ReDRAW | 可用于 fidelity 或 shift，不自动改善控制 |

在**固定、reward-free、offline trajectories**这一主协议中，可合法使用的主要是行为策略下的
TD-style predictive target、自监督 action/occupancy signal 和 held-out data calibration。
它们通常更准确地叫 representation/world-model learning，而不是 online RL。若加入环境
reward、主动采样或 imagined actor-critic，就必须更换协议并与对应 MBRL/WAM baseline 比较。

本文常用缩写：MBRL 是 model-based RL；MPC 是 model predictive control；CEM/MPPI 是
两类采样规划器；JEPA 是 joint-embedding predictive architecture；SR/SF 是 successor
representation/features；GHM 是 generative horizon model；OOD 是分布外；BFM 是 behavior
foundation model。**BC 是 behavioral cloning（行为克隆）**：把数据中的 `(observation,
action)` 当监督样本模仿行为策略，本身不做 reward maximization，也不等同于 online RL。

## 3. 领域演进图谱

### 3.1 经典显式动力学与不确定性

Dyna 在 1990 年已经把 real experience、learned model、planning update 和 acting 放进同一
循环，是“RL 帮助或使用 world model”的历史根。现代经典 MBRL 学习 `p(s'|s,a)`，随后用
trajectory optimization 或 policy search 控制。PILCO 代表 Gaussian-process 路线；PETS
使用 probabilistic ensembles 和 trajectory sampling，把 epistemic 与 aleatoric uncertainty
带入 MPC；MBPO 发现从真实数据状态出发的短 imagined rollouts 能在偏差和数据增益之间取得
更稳妥的折中。Plan2Explore 则用 ensemble disagreement 作为 imagined novelty，主动采集
更有利于 task-agnostic world model 的数据，说明 RL 还可以通过**改变数据分布**帮助模型。

这条路线给本项目三条长期有效的教训：

- one-step supervised accuracy 不会自动变成长时 control accuracy；
- 模型被 planner 主动查询后会遇到训练分布外动作，必须报告 action support；
- uncertainty、rollout horizon 和 replanning frequency 与模型本身同等重要。

### 3.2 从像素学习 stochastic latent state-space model

World Models（2018）展示了在压缩视觉 latent 中训练 controller，并可在 imagined world
中学习。PlaNet 用 RSSM 和 latent overshooting 从像素学习 belief/dynamics，再用 CEM
规划。Dreamer 系列将主要用途改为 latent imagination 中的 actor-critic 学习，DreamerV2
使用离散 latent，DreamerV3 则以单一配置扩展到大量 domain。Dreamer 4 进一步以 shortcut
forcing 和高效 Transformer 扩展到 Minecraft，从少量 action-labelled data 学动作条件、从
大量无标注视频学环境知识，并在纯 offline imagination 中解决超过两万步动作的 diamond
任务。DayDreamer 又把这条路线带到真实机器人。

SimPLe 直接用 learned video model 在低数据 Atari 中训练 policy；I2A 不规定 planner，而让
policy 学会解释 imperfect model rollouts；SLAC 学 stochastic latent variable model 来提供
policy state，但不在模型里 rollout policy。这三者分别代表 synthetic environment、learned
imagination features 与 representation-only use，提醒我们“训练了 dynamics”不等于同一种
model-based usage。IRIS、TWM、STORM 等 Transformer world models 则强化长序列建模和离散
token 生成。
它们回答的是“怎样在 learned simulator 内训练 policy”，与 LeWM 的 reward-free
image-goal MPC 并不属于同一评测协议，但构成 world-model RL 的主干，不能在相关工作中
只列 TD-MPC2 而忽略。随机和多模态未来也不能被 deterministic latent MSE 代替；WIMLE
代表显式 stochastic multi-modal dynamics、ensemble uncertainty 和 confidence-weighted
synthetic transitions 的路线。

Reconstruction-free MBRL 也不是 LeWM 才出现。DreamerPro 用 prototype assignment 取代
像素重建，并把 recurrent temporal state 蒸馏进 prototype；MuDreamer 以 value 和 previous
action prediction 取代 reconstruction，并用 batch normalization 防 collapse；R2-Dreamer
用 redundancy reduction 作为内部正则，在不依赖 decoder 或 data augmentation 时防止
collapse；Dreamer-CDP 则把 JEPA-style continuous deterministic prediction 接入 Dreamer。
因此，“decoder-free + RL + anti-collapse”已经是明确竞争线，而不是待填空白。

### 3.3 Task-oriented implicit world model

MuZero 不重建 observation dynamics，而学习对 search 有用的 latent transition、reward、
policy 和 value。TD-MPC 及 TD-MPC2 面向连续控制，将局部 latent dynamics、reward、Q、
actor 与 MPC 结合。这些方法证明“RL 信号和 latent prediction 联合训练”早已成立。

但需要精确表述梯度路径：TD-MPC 的 latent consistency 直接训练 encoder/dynamics，TD
loss 主要训练 Q；整个模型联合优化不等于 Q loss 本身必然校准了每个 transition direction。
因此本项目可以研究 RL 信号是否进入 LeWM dynamics，但不能以“第一次让 TD 帮助 world
model”为贡献。

### 3.4 Offline model-based RL

MOPO、MOReL、COMBO 和 RAMBO 研究固定数据下的 model exploitation。它们分别通过
uncertainty penalty、pessimistic MDP、conservative value learning 或 adversarial model
避免 policy 利用模型在 dataset support 外的错误。

本项目同样使用离线 trajectories。任何 learned actor、successor policy 或 MPC search
产生的 OOD action 都可能让 LeWM 与 critic 相互确认错误。因此，仅使用 reward-free 数据
并不自动使方法安全；coverage、conservatism 和 model disagreement 是必需诊断。

### 3.5 Decision-aware、value-aware 与 policy-aware model learning

这一研究线直接指出 world-model objective mismatch：最低 one-step likelihood/MSE 不一定
带来最高 return。主要工作包括：

| 工作 | 核心思想 | 对本项目的约束 |
| --- | --- | --- |
| VAML / IterVAML | 用 value/function class 定义 model error | 普通 value loss 不是创新 |
| PAML | 根据 policy-gradient planner 的使用方式训练模型 | planner-aware model loss 已存在 |
| Minimax Model Learning | 从 off-policy evaluation/optimization 推导 model loss | 固定离线数据下已有决策导向理论 |
| VaGraM | 用 value gradient 重加权 state prediction error | “控制相关方向优先”已有直接实现 |
| TOM | 匹配真实与模型内 current-policy transition occupancy | 与 model-vs-data occupancy 很接近 |
| Value Equivalence | 只要求模型对函数/策略集合产生相同 Bellman updates | 提案的理论语言并非新概念 |
| Calibrated VAML | 指出常用 value-aware surrogate 可能不校准 | loss 下降不能直接推出模型更好 |

因此，原先“让预测 transition 与真实 transition 在 Bellman functional 下相等”的单步
公式应降级为 baseline。它既可能只是 VAML/MuZero surrogate，也可能由 encoder 与 value
head 一起漂移形成低 loss、错误 dynamics 的解。

另一个相邻分支是 Predictron、Model-Based Value Expansion（MVE）和 STEVE：用短期模型
rollout 加 terminal value，或按模型不确定性混合不同 rollout horizon。它们未必联合校准
LeWM 与 successor operator，但已覆盖“局部模型负责近期、value 负责远期”的系统结构。
因此，给 LeWM 增加 terminal value 只能算 planner baseline，不能证明 world model 被改善。

### 3.6 Control-sufficient representation

DeepMDP、DBC、MICo、PSE 与 BS-MPC 用 reward、transition、behavioral similarity 或
bisimulation 塑造控制相关 latent。这条路线的核心不是复原全部 observation，而是保留使
value/policy/transition 可迁移的等价类。C-SWM、Structured World Belief、Graph Networks
as Learnable Physics Engines、Causal-JEPA、Dyn-O 和 FIOC-WM 则加入 object、relation、
interaction 或 belief structure，提升组合泛化和可解释性。

更早的 Predictive State Representation 直接用在 action-observation tests 下对可观察未来的
预测定义 state；Causal State Representation 寻找 joint action-observation histories 的最粗
充分划分；Causal World Models 则显式估计 latent confounder，以支持 intervention 和
counterfactual prediction。另一条 operator 路线使用 Koopman lifting，把 nonlinear dynamics
近似为 observable space 中的线性演化，并与 Bellman/HJB 或 offline RL 结合。这些工作提醒
我们：有物理或控制意义的 state 不一定等于 encoder 的单帧 embedding，也可能是 history
equivalence class、predictive tests、objects 或受约束 operator。

这说明“物理意义”必须拆成可测性质：state identifiability、action consequence、object
permanence、counterfactual separation、equivariance、causal intervention、transition mode
或 planning transfer。仅展示 PCA、linear probe 或漂亮 rollout 不足以证明模型理解物理。

### 3.7 Successor representation 与长期 predictive world model

Successor representation（SR）缓存 policy 下的 discounted future occupancy：

```math
M^\pi(s,a,s') = \mathbb E_\pi\left[
\sum_{k\ge 0}\gamma^k \mathbf 1(s_{t+k+1}=s')
\mid s_t=s,a_t=a\right].
```

Successor features 将 state indicator 换为 feature `phi(s)`，并可在 reward 线性分解时快速
重估 value。Forward-Backward（FB）、HILP、FRE 和 One-step FB 将这条路线扩展到
reward-free zero-shot RL；TD-JEPA 从离线、无奖励 transition 学 policy-conditioned 的
multi-step latent predictor、task encoder 和 policies；RLDP 则表明带 orthogonality 的
latent dynamics prediction 本身就是很强的 zero-shot representation baseline。

更关键的是，`gamma`-models 已把 TD 解释为学习无限概率时域的 generative dynamics；
TD-Flow 用 probability-path Bellman equation 和 flow matching 改善长期预测；Universal
Horizon Models 直接预测任意 horizon；Jumpy World Models 学预训练 policy 所诱导的多尺度
occupancy，并用跨尺度 consistency 做组合规划。

这组工作实质上削弱了原方案剩余的新颖性。所谓 local–successor consistency 若只是让
一步 rollout 的累计 features 接近 successor features，很可能是已有 one-step model、SR、
`gamma`-model 或 multi-timescale consistency 的另一种参数化。

### 3.8 Reconstruction-free visual world model 与 JEPA

DINO-WM 使用 frozen DINOv2 patch features 学 action-conditioned predictor；PLDM 从头
训练 JEPA encoder/predictor，并依赖 VCReg、temporal smoothness 和 inverse dynamics；LeWM
以 next-embedding prediction + SIGReg 实现较简洁的端到端稳定训练。V-JEPA 2-AC 则展示
互联网视频预训练加少量无标注机器人视频可以支持 real-robot image-goal planning。

这条路线应追溯到 V-JEPA 的 feature-space masked video prediction，以及 LeJEPA 对 SIGReg
的系统化。防 collapse 的机制至少有四族：target asymmetry/stop-gradient、negative samples、
variance-covariance 或 redundancy reduction、以及 reconstruction/physical grounding。
Self-Predictive RL 还给出了 stop-gradient 与 state/history abstraction 的理论联系。它们解决
的是优化退化，不自动保证 latent 对 partial observability、counterfactual actions 或 task
relevant variables 可辨识。

2026 年 LeWM/JEPA 的直接改进非常密集：

| 工作 | 主要问题 | 机制 | 对本项目的影响 |
| --- | --- | --- | --- |
| Value-guided JEPA | Euclidean latent distance 与 goal value 不一致 | value/quasi-distance shaping | `LeWM + goal value` 已被覆盖 |
| Reward-free bisimulation JEPA | slow visual features/distractors | transition-behavior equivalence | reward-free control-aware JEPA 已存在 |
| Temporal Straightening | latent trajectory curvature 妨碍优化 | curvature regularization | planning geometry 已是独立路线 |
| RC-aux | predictive but not plannable | multi-horizon、reachability、negatives | 长期/可达辅助已直接用于 LeWM |
| Fast-LeWM | autoregressive drift 与规划速度 | action-prefix multi-horizon prediction | 多时域直接预测已覆盖 |
| Hierarchical Planning / Hi-LeWM | 搜索深度和 subgoal | 多尺度模型或 frozen LeWM 上层规划 | hierarchy 不等于改进 dynamics |
| AdaJEPA | test-time dynamics shift | 新 transition 上自监督适配后 replanning | transition revaluation 不是新点 |
| Temporal-Distance JEPA | latent cost 不表示进展 | directed temporal cost + rollout consistency | reward-free plan-aware cost 已覆盖 |
| Centered SIGReg | 多任务 latent cluster aliasing | 对 temporal residual 做 SIGReg | anti-collapse 叙事需考虑 multi-task 结构 |
| PhyLatent | physical/counterfactual collapse | state grounding、alignment、branch separation | “首次物理有意义”不可声称 |
| PSG-JEPA | robot state/change 不可辨识 | proprioception 与 joint-change grounding | 使用物理监督的强上限 |
| Metric Non-Collapse | folding 与 metric fidelity | local-global metric hinge + transfer theory | “防坍塌保证规划”已有新理论前身 |

### 3.9 Foundation/video world models 与 RL post-training

UniSim、Genie、Cosmos、V-JEPA 2 和 iVideoGPT 等把 world model 扩展到大规模
action-conditioned video simulation、latent actions 和多数据源预训练。iVideoGPT 把
observation、action、reward 统一 token 化，并在大规模 human/robot trajectories 上预训练，
再适配 video prediction、planning 和 MBRL。DIAMOND、GameNGen 与 Genie 则代表 diffusion
或 generative interactive environments：它们可以实时或近实时模拟游戏，但短视频感知质量
不能代替长期状态、action adherence 和 policy utility 评测。Dreamer 4 正在缩小 scalable
video world model 与 imagination RL 之间的界线。

更直接地，RLVR-World 已用 verifiable rewards 后训练语言和视频 world models；RLIR 用
inverse-dynamics action recovery 构造视频 action-following reward；WorldCompass 和
Persistent Robot World Models 用 RL 改善长时交互/视频 rollout。这意味着“用 RL 直接训练
world model generator”也不是空白。它们与 LeWM 的差别在于 reward 主要衡量 decoded
prediction 或视觉/action fidelity，而不是下游 control sufficiency，但必须纳入领域定位。

### 3.10 Adaptation、continual world model 与 dynamics shift

World model 的另一条主线不是预训练后冻结，而是进入新动力学后持续校准。Finetuning
Offline World Models in the Real World 研究 offline pretraining 到 online finetuning；AugWM
针对单一离线环境做 zero-shot dynamics generalization；ReDRAW 直接用 latent-state dynamics
residual 修正 sim-to-real world model；SPREAD 做 online continual adaptation；AdaJEPA 则在
每次 MPC 执行得到新 transition 后自监督更新再规划。

这组工作进一步压缩候选空间：**轻量 residual adapter、test-time self-supervised update、
transition shift 适配本身都不是创新点。**它们可以成为实现手段或强 baseline。候选贡献若
存在，必须来自 local 与 temporally abstract model 的新关系及可归因证据，而不能来自“用了
adapter”或“看了少量目标域 transition”。

### 3.11 World Action Model、latent action 与机器人 policy

机器人领域正在把“预测未来”和“输出动作”合并。Video Prediction Policy 用 video
diffusion 内部的 future representations 条件化 inverse dynamics；DreamGen 用 video world
model 生成 neural trajectories，再用 latent action 或 inverse dynamics 补动作；VeoRL 从
无标注视频构建 interactive world model，帮助 offline RL。DreamZero 把 pretrained video
diffusion 改造成同时预测 video 与 action 的 World Action Model，并用于 closed-loop
zero-shot policy。

与 LeWM 更接近的是 latent-action 路线。Learning Latent Action World Models In The Wild
从无标签野外视频学连续、受约束 latent actions，并通过已知 action 到 latent action 的
controller 接回规划；LaWAM 不解码视频，而让 latent-action-conditioned model 预测 latent
visual subgoals，再条件化 action generation。它说明“latent future 比 pixel video 更高效，
且能帮助 policy”已经有直接实现。VLA-MBPO 则反向把 pixel world model 当 RL 环境训练 VLA，
并用 chunk-level branched rollouts 缓解误差累积。

这些方法大多优化 direct policy 或 action chunk，而 LeWM 优化任意 primitive action
sequence 的 MPC。两者接口不同，但如果本项目引入 actor、inverse dynamics、latent action
或 joint future-action head，它们就是必须比较的最近邻，不能只对比 TD-JEPA。

### 3.12 显式空间、3D/4D physics 与自动驾驶

world model 的空间表示不只有 global latent vector。object slots/graphs、spatial latent grid、
BEV/occupancy、point cloud/particles、NeRF 和 Gaussian Splatting 都可以承担 state。HD-VPD
和 ParticleFormer 用大规模 3D particles/point clouds 表达多物体、多材料动力学；PIN-WM
将 differentiable rigid-body physics 与 Gaussian Splatting observation loss 结合；Physically
Controllable World Model 则以可条件查询的概率模型学习 objects、appearance 与 dynamics。
这条线对“真正物理意义”给出更强、也更昂贵的对照：结构可解释性通常来自明确 spatial
inductive bias 或 physical variables，而不只是更换 latent regularizer。

自动驾驶形成了另一套重要谱系。MILE 在 camera-only offline data 上联合 world model 与
driving policy，并以 3D geometry 和可解码 BEV 为归纳偏置；DriveWorld 用 4D scene
pretraining 服务驾驶感知；GAIA-1 和 DriveDreamer 代表 action-conditioned driving video
generation；Latent-WAM 则在 multi-view compact scene tokens 中预测未来 world status 并
直接用于 trajectory planning。这些工作验证 spatial geometry、multi-view consistency 和
实时性的重要性，但数据、action、state 与指标均不同，不能作为 Stable World Model 四任务
上的同表 baseline，只能作为结构设计和外部效度证据。

## 4. World Model 的核心矛盾

### 4.1 Prediction objective 与 decision objective 不一致

模型可能在大面积无关区域预测很好，却在 planner 真正访问的小区域犯关键错误；反过来，
MuZero 式 task-oriented model 可能 observation fidelity 很差却支持正确决策。因此必须把
下列问题分开：

- 是否预测了真实未来；
- 是否对特定 policy/reward 保持 value；
- latent geometry 是否能作为规划 cost；
- planner 是否能在给定预算中找到好动作；
- actor/critic 是否利用了模型漏洞。

### 4.2 局部可滚动性与长期抽象的权衡

一步模型 policy-independent、可组合任意 action sequence，对 transition revaluation 灵活，
但推理时反复 rollout 会累计误差。SR/GHM/TD-Flow 直接缓存长期未来，推理更快、避免逐步
累积，却依赖 policy 或预训练 skill，面对 transition/policy 变化需要更新。多尺度 world
model 尝试取中间地带，但必须保证各尺度预测一致。Simulation Lemma 一类结果也从 value
error 角度说明，小的 transition misspecification 会随有效 planning horizon 放大；因此不能
用 one-step loss 代替 long-horizon 与 decision-level 证据。

### 4.3 通用预测与任务充分性的权衡

重建或自监督 prediction 信号丰富、可跨任务复用，但可能浪费容量在背景纹理；reward/value
监督聚焦决策，却可能过拟合单任务并丢失新任务需要的信息。真正的 reward-free generality
不能靠一句“没有环境 reward”保证，而取决于 feature/task basis 覆盖了哪些未来 reward。

### 4.4 非坍塌不等于可辨识或可规划

latent 有满秩协方差，不代表不同物理状态可区分、动作后果正确、距离单调或 counterfactual
分支分离。SIGReg、orthogonality、whitening 只解决统计退化的一部分；physical grounding、
metric structure、temporal order 和 causal intervention 是不同性质。

反过来，加入 reward/TD 也不能保证避免 collapse。稀疏 reward 可能只保留极窄 task
subspace；joint encoder、value 和 model 可能协同漂移；policy-conditioned target 还可能把
behavior policy 的偏差写进表示。必须分别测试 statistical collapse、physical aliasing、
action marginalization 和 task over-specialization。

### 4.5 Offline coverage 与 model exploitation

在固定数据上，planner/actor 会查询行为数据未覆盖的 action sequences。世界模型、successor
head 与 policy 联合优化时尤其可能形成自洽但错误的闭环。任何增益都要检查 dataset support、
ensemble disagreement、conservative constraints 及真实 transition 上的 held-out error。

### 4.6 Evaluation coupling

最终 success 同时由 encoder、dynamics、cost/reward、solver、terminal value、policy prior、
replanning 和 compute 决定。只报 success 无法判断改进发生在哪。相同 world model 换 solver
即可产生巨大差异，Stable World Model 的 LeWM 消融也显示 CEM 与 gradient solvers 差距
明显。因此主实验必须固定 planner，并把更强 cost/value 作为单独系统上限。

### 4.7 多模态未来、action identifiability 与 hallucination

相同历史和动作可能对应多个合理未来。deterministic MSE 会平均模式，generative model 又
可能生成感知上流畅但动力学错误的未来。没有 action labels 时，latent action 还可能把
camera motion、外部 agent 或环境噪声误认成 controllable action。MMBench2 将生成式 world
model 的错误区分为 perceptual、action-marginalized 和 scene-diverging hallucination，并将
其主要关联到 state-action coverage。因此候选方法除 long-horizon error 外，还必须测
action adherence、多模态 calibration、coverage 和 closed-loop divergence。

## 5. 最接近候选想法的工作矩阵

候选想法暂称 **primitive local model 与 policy-conditioned horizon model 的协同学习**，
不是正式方法名。

| 方法 | Offline reward-free | Pixels | Primitive arbitrary-action rollout | Policy-conditioned long horizon | 跨尺度/模型一致性 | Test-time use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LeWM | 是 | 是 | 是 | 否 | 否 | MPC |
| PLDM / DINO-WM | 是 | 是 | 是 | 否 | 否 | MPC |
| TD-MPC2 | 否 | 可 | 是 | Q/actor | joint losses | MPPI/CEM + Q |
| DreamerPro / R2-Dreamer | 否 | 是 | policy rollout | actor-critic horizon | SSL + latent dynamics | imagination RL |
| Dreamer 4 / iVideoGPT | mixed | 是 | action-conditioned simulator | 长视频/agent horizon | sequence model | imagination RL / planning |
| VAML / TOM | 依实现 | 可 | 是 | value/occupancy | model-data decision objective | policy/MPC |
| MVE / STEVE | online 或 replay | 可 | 短 rollout | terminal value | 按 horizon/uncertainty 组合 | policy/value learning |
| FB / One-step FB | 是 | 可 | 否 | 是 | successor factorization | direct policy |
| TD-JEPA | 是 | 是 | 否 | 是 | TD latent prediction | direct zero-shot policy |
| RLDP | 是 | 可 | representation stage | 后接 BFM | dynamics + anti-collapse | direct policy |
| `gamma`-model / TD-Flow | 是或 off-policy | 主要 state | 否 | 是 | TD distribution recursion | value/planning |
| UHM | offline | 主要 state | 否 | 任意 horizon | horizon-conditioned | offline value learning |
| Jumpy World Models | 是 | 主要 state | policy/skill sequence | 是、多尺度 | 明确跨尺度 consistency | policy composition |
| Fast-LeWM | 是 | 是 | action-prefix | 否 | 多 horizon supervision | MPC |
| RC-aux / Temporal-Distance | 是 | 是 | 是 | reachability/temporal cost | rollout-aligned auxiliary | MPC |
| ReDRAW / AdaJEPA | source pretrain + 少量 target transition | 是 | 是 | 否 | residual 或 online recalibration | adaptation + control |
| DreamZero / LaWAM | offline mixed robot data | 是 | direct action/chunk | future video 或 latent subgoal | joint world-action modeling | direct policy |
| Latent Action WM | video-only 可预训练 | 是 | 需 action mapper | latent-action-conditioned | inverse/forward consistency | planning / policy |
| 候选问题 | 是 | 是 | 是 | 是 | local rollout 对 data-derived horizon operator | MPC，必要时 direct policy |

若候选保持纯 MPC，最危险的近邻不是 TD-JEPA，而是 **Jumpy World Models + TD-Flow/UHM +
VAML/TOM** 的组合；若加入 actor、latent action 或 action generation，**LaWAM/DreamZero**
立即成为同等重要的近邻。
候选工作必须逐条回答：

1. 为什么需要显式 primitive LeWM，而不是直接使用 GHM/UHM/Jumpy model？
2. 所谓 consistency 是否超出 Jumpy World Models 的 cross-timescale consistency？
3. data target 是 value、feature expectation、density/flow 还是 occupancy ratio？
4. 为什么它比 multi-horizon prediction 或 Temporal-Distance rollout consistency 更有信息？
5. 改进是否发生在 encoder/dynamics，而不是新 terminal value、policy 或 planner cost？
6. 若使用 latent action 或 policy head，为什么不是 LaWAM/DreamZero 的较小规模变体？

在这些问题回答前，不应锁定模型架构或命名。

## 6. 研究判断：哪些方向已经不值得主打

| 原始主张 | 判断 | 原因 |
| --- | --- | --- |
| RL 帮助 LeWM | 太宽泛 | MBRL、TD-MPC、Dreamer、RLVR-World 都已覆盖 |
| TD 防止特征坍塌 | 不足以创新 | Self-Predictive RL、TD-JEPA、RLDP、SIGReg 后续已覆盖 |
| decoder-free MBRL 防坍塌 | 不新 | DreamerPro、MuDreamer、R2-Dreamer、Dreamer-CDP 已直接覆盖 |
| 首次得到物理 latent | 不成立 | LeWM probes、PhyLatent、PSG-JEPA、物理结构 WM |
| 用 value 校准 dynamics | 不新 | VAML、VaGraM、Value Equivalence、TOM |
| reward-free reachability | 不新 | HILP、RC-aux、Temporal-Distance JEPA |
| 一步 + 长期 model | 高度拥挤 | SR、`gamma`-model、TD-Flow、UHM、Jumpy WM |
| transition revaluation | 不能独立成贡献 | model-based/SR 文献与 AdaJEPA 已直接研究 |
| latent dynamics residual adapter | 不新 | ReDRAW 已直接用于视觉 world-model adaptation |
| 联合预测未来与动作 | 不新 | VPP、DreamZero、LaWAM 和 latent-action WM 已覆盖 |
| 视频逼真证明可作为 simulator | 证据不足 | WorldModelBench、WorldBench、WorldGym、MMBench2 均显示缺口 |
| 更高 success 就是更好 WM | 错误归因 | success 与 cost、solver、value、policy 强耦合 |

## 7. 当前仍可检验的候选问题

### 7.1 最小科学问题

与其一开始设计复杂 joint actor，不如先问：

> 一个从真实数据学习的 policy-conditioned horizon operator，是否能提供 LeWM 多步 rollout
> error 中普通 latent MSE 看不到、且对 held-out planning 有预测力的 residual signal？

这一步只需要冻结或弱更新 LeWM，不需要宣称 zero-shot RL 新算法。候选统计量可以是同一
feature map 下的 discounted feature expectation、distributional occupancy discrepancy 或
horizon-conditioned future embedding error。具体形式必须在复现 Jumpy WM/TD-Flow 后确定。

### 7.2 只有满足这些条件才可能形成方法贡献

1. target 来自真实 held-out transitions 或其稳定估计，不是 model 自举自身；
2. local model 能继续接收 arbitrary primitive action sequence，不退化成 policy-only model；
3. 比 one-step/multi-horizon latent prediction、VAML/TOM surrogate、Jumpy consistency 都强；
4. 在 frozen planner cost 下提升，而不是依赖 successor terminal value；
5. 在 reward revaluation 与 transition revaluation 中呈现可解释的互补优势；
6. 在 held-out actions 上保持 action adherence，不产生 action-marginalized hallucination；
7. 对 encoder、dynamics、cost 和 policy 的梯度路径有明确归因。

### 7.3 应优先探索的三条窄路线

**路线 A：Horizon-conditioned diagnostic，不联合训练。** 在 frozen LeWM 上学习多个 horizon
的 data operator，用它预测哪些 candidate rollouts 会失败。若该误差与控制失败无关，整个
方向可以快速终止。

**路线 B：Detached calibration adapter，作为受控探针而非贡献。** 先只训练 dynamics
residual/adapter，不让 TD gradient 进入 encoder，避免破坏 SIGReg latent。ReDRAW 是必须
比较的直接 baseline；只有超出普通 residual adaptation，才测试 encoder co-training。

**路线 C：Uncertainty-aware arbitration。** local rollout 与 horizon model 不强行相等，
而在二者高置信一致时规划，在分歧时缩短 rollout、replan 或拒绝 OOD action。这比简单
consistency loss 更贴近 offline model exploitation，但也必须与 ensemble/pessimistic MBRL
比较。

## 8. 必须采用的实验协议

### 8.1 协议 A：固定 reward-free 数据的 image-goal MPC

训练不使用环境 reward；测试给 goal image。所有方法共享 dataset split、frame stack、
encoder resolution、CEM/MPPI、horizon、candidate count、iterations、replanning period 和
wall-clock/forward-pass budget。

P0 baselines：LeWM、PLDM、DINO-WM pixels-only、Fast-LeWM，以及 frozen LeWM + 等参数
auxiliary。能公平接入时加入 RC-aux、Temporal-Distance JEPA、Temporal Straightening。

### 8.2 协议 B：Zero-shot reward transfer

训练 world/representation 时无环境 reward；测试 reward 必须未见，并为所有方法提供同量
reward inference samples。P0 baselines：frozen LeWM + same BFM stack、RLDP、TD-JEPA、
FB、One-step FB、HILP、FRE。Direct policy 与 MPC 分开报告，不做混合排行榜。

### 8.3 协议 C：Reward-labelled control

TD-MPC2、Dreamer、R2-Dreamer 和 value-aware model losses 需要 reward，应单独比较。
只有 reward labels、online interaction、replay、updates 和 evaluation budget 相同，才可
作系统结论。Dreamer 4 的数据规模和任务跨度不同，只作 foundation-scale evidence，除非能
构造 matched-data、matched-compute 配置，否则不能放入本项目主排行榜。

### 8.4 协议 D：Long-horizon predictive models

比较 LeWM rollout、multi-horizon predictor、`gamma`-model/TD-Flow/UHM 和 Jumpy model。
state-based 与 pixel-based 结果分层报告；同一 policy family、horizon distribution、数据和
future-state metric 下比较。该协议先回答候选 operator 是否真的更好，不急于接入控制。

### 8.5 协议 E：Revaluation 与 adaptation

- reward revaluation：transition 不变，仅改变 reward；
- policy revaluation：改变 action constraint 或 policy family；
- transition revaluation：改变质量、摩擦、障碍、接触或 actuator response；
- joint revaluation：reward 与 dynamics 同时改变。

必须明确测试时能看到多少新 transitions。transition adaptation 对照至少包含 frozen model、
successor-only update、LeWM local update、AdaJEPA 式 test-time update 和候选方法。
若设置包含 source-to-target dynamics shift，还必须加入 ReDRAW 式 latent residual adaptation。

### 8.6 协议 F：World Action Model 与 action interface

只有候选引入 actor、inverse dynamics、latent actions 或 action chunks 时才启用。比较至少
包含 BC/GCBC、frozen world model + same policy head、VPP-style future feature policy、
latent-action model 和 LaWAM-style latent subgoal policy。必须分别报告：

- action-labelled、video-only 和 mixed-embodiment 数据量；
- primitive action、latent action 与 action chunk 的控制频率和 horizon；
- open-loop action prediction 与 closed-loop success；
- latency、replanning rate、action adherence 和 cross-embodiment transfer。

Direct policy、MPC 和 world-model environment 内训练的 policy 属于三种不同用法，不混成
同一排行榜。DreamZero/VLA-MBPO 只能在输入、动作接口和数据预算可比时作直接 baseline。

## 9. 归因矩阵

| 编号 | 设置 | 回答的问题 |
| --- | --- | --- |
| A0 | LeWM | 原始基线 |
| A1 | frozen LeWM + detached horizon head | latent 本身是否已经够用 |
| A2 | frozen LeWM + same policy/value/planner | 收益是否只来自下游模块 |
| A3 | LeWM + multi-horizon prediction | 是否只是更长 prediction supervision |
| A4 | LeWM + one-step value/VAML surrogate | 是否只需已有 decision-aware loss |
| A5 | LeWM + Jumpy/TD-Flow-style horizon model | 最近长期模型基线 |
| A6 | gradient 仅进 adapter/dynamics | 是否需要改 encoder |
| A7 | gradient 进入 encoder + dynamics | 完整 co-training 是否必要 |
| A8 | 仅 successor terminal cost | 是否只是 planner enhancement |
| A9 | matched random auxiliary / extra parameters | 是否只是容量和正则收益 |
| A10 | equal compute and update counts | 是否只是更多训练 |
| A11 | shuffled/zeroed actions 与 held-out action branches | 模型是否真正使用 action |
| A12 | matched redundancy-reduction regularizer | 收益是否只是 R2-Dreamer 式 anti-collapse |

主结论必须来自 A0、A3、A4、A5、A6、A7、A11、A12 在**相同原始 planner cost**下的
比较。A8 只能作为系统上限单列。

## 10. 评价指标

### 10.1 World-model fidelity

- one-step latent prediction；
- 5/10/20/50-step open-loop error，按 horizon 作曲线；
- posterior-free rollout 与 teacher-forced rollout 的 gap；
- action sensitivity、counterfactual branch separation；
- action adherence，以及 perceptual/action-marginalized/scene-diverging failure rate；
- stochastic calibration、mode coverage 与多样性，不只报平均预测；
- calibrated uncertainty、ensemble disagreement 与 OOD detection；
- held-out policy 下的 future-state/occupancy/operator error；
- local rollout 与 horizon model 的 disagreement。

### 10.2 Representation

- covariance spectrum、effective rank、pairwise cosine similarity；
- state/proprioception 的 linear 与 nonlinear probes；
- latent pair 对 physical change 的可辨识性；
- visual distractor invariance 与 hidden-dynamics aliasing；
- temporal ordering、reachability ranking 和 metric distortion；
- task/reward feature span 与 held-out reward approximation error。

### 10.3 Decision utility

- fixed-cost MPC success 和逐 episode outcome；
- planner regret 或 candidate-ranking accuracy；
- policy-value correlation、policy-ranking agreement 与 ID/OOD calibration；
- 用模型训练或选择 policy 后，相对不用模型的 optimization lift；
- adversarial/exploitability audit：policy 是否主动寻找模型漏洞；
- trajectory stitching；
- reward、policy、transition revaluation；
- planning latency、model calls、参数量、显存和 wall-clock；
- failure attribution：model、cost、search、value、policy 或 data support。

### 10.4 从“像”到“可用于决策”的证据阶梯

评价不能只在一个数字上完成。参考 decision-making-centric 的 L0-L7 思路，本项目采用下列
逐层证据，后一层不能由前一层自动推出：

| 层次 | 本项目要求的证据 | 能支持的最强表述 |
| --- | --- | --- |
| E0 | 可视化、latent rollout sanity check | 模型能生成或滚动 |
| E1 | pixel/feature/state prediction、physics probes | 对被测变量有预测 fidelity |
| E2 | held-out action sensitivity、action adherence | 学到了动作干预效应 |
| E3 | closed-loop long rollout、OOD 和 uncertainty calibration | 在被测分布上可充当有限 simulator |
| E4 | reward/value error 与 policy-ranking agreement | 可辅助 policy evaluation |
| E5 | fixed-budget planning success、candidate ranking | 可辅助 planning |
| E6 | policy optimization lift 与 exploitability audit | 可作为 policy learning environment |
| E7 | 新任务/新动力学/真实系统 transfer | 有超出 benchmark 的决策效度 |

WorldModelBench/WorldBench 主要补 E1 的物理诊断；WorldGym/WorldEval 触及 E2-E4；真正
声称“RL 改善了 LeWM”至少需要 E1、E2、E5 同时改善，声称“可在模型内训练 policy”还需要
E6。视觉 plausibility、effective rank 或单一 success rate 都不能单独越级。

### 10.5 证据口径

作者报告、第三方 checkpoint、本项目复现必须分开。Stable World Model 当前公开四任务
结果为：

| 方法 | Two-Room | Reacher | PushT | OGB Cube | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLDM | 97 | 78 | 78 | 65 | 79.5 |
| DINO-WM pixels-only | 100 | 79 | 74 | 86 | 84.8 |
| LeWM | 87 | 86 | 96 | 74 | 85.8 |

这些是上游当前页面的公开口径，不是本项目复现。Two-Room 接近饱和；PushT 的 LeWM
已经很高，适合不退化检查；Cube 能暴露表示差异，但 random success 较高；Reacher 适合
快速稳定性迭代。正式声称提升前必须至少多 seed，并保存逐 episode 结果。

## 11. 最小可行研究路径

### 阶段 0：复现平台与 baseline

使用 `stable-worldmodel[all]==0.1.1` 的公开 API，先复现 LeWM checkpoint evaluation，
再在相同 pipeline 下接 PLDM/DINO-WM。不得复制或修改上游 baseline 源码。

### 阶段 1：Frozen representation audit

冻结 LeWM，训练 RLDP/FB/TD-JEPA-compatible heads 与 horizon operator，评估 held-out
reward fitting、operator error 和 candidate failure prediction。如果 frozen LeWM 已达到相同
结果，则 RL gradient 无需进入 world model。

### 阶段 2：Long-horizon null hypotheses

实现或调用已发布的 multi-horizon、TD-Flow/UHM/Jumpy baseline。在 state-based 小任务上
先验证公式，再进入 pixels。若候选 statistic 不超过现有长期模型，停止。

### 阶段 3：最小 adapter 实验

只允许 gradient 进入轻量 dynamics adapter；固定 encoder、planner、cost 和 compute，并
与 ReDRAW 式 residual 适配比较。只有 model fidelity 与 fixed-cost MPC 同时改善，且不能
由普通 residual adaptation 解释，才继续端到端训练。

### 阶段 4：End-to-end attribution

逐步开放 encoder gradient，并跟踪每个 loss 在 encoder/dynamics 上的 gradient norm 与
cosine similarity。加入 physical probes、rank、OOD support 和多 seed。

### 阶段 5：Revaluation

分别做 reward 和 transition revaluation，验证 local model 的灵活性与 cached horizon model
的效率是否真有互补，而不是在 IID task 上追求小幅平均分。

## 12. Go / No-Go 标准

继续形成论文方案需要同时满足：

1. 在 held-out policies/horizons 上，候选校准显著降低真实 long-horizon operator error；
2. fixed planner、fixed cost、fixed compute 下提升不少于两个非饱和任务；
3. frozen LeWM + same stack、RLDP、multi-horizon prediction、VAML surrogate 和 Jumpy/
   TD-Flow baseline 不能解释增益；
4. 一步预测、长时 rollout、latent rank 与 physical probes 没有明显退化；
5. held-out actions 上的 adherence、closed-loop divergence 和 uncertainty 没有退化；
6. 至少在一种 reward/transition revaluation 中展示局部模型和长期模型的互补性；
7. 多 seed 置信区间支持结论。

满足以下任一项则停止或改题：

- frozen LeWM/RLDP 已获得全部收益；
- R2-Dreamer 式 redundancy reduction 已获得全部 anti-collapse 收益；
- 只有新 value/cost head 提升 success；
- Jumpy WM、UHM、Fast-LeWM 或普通 multi-horizon loss 达到相同结果；
- ReDRAW 式 residual adapter 已解释适配收益；
- method 只在单一 reward 或饱和 benchmark 有效；
- successor/model joint training 在低 coverage 下不稳定或利用 OOD actions；
- 增益伴随 action marginalization、mode averaging 或 policy-ranking 失真；
- 公式与现有 cross-timescale consistency 没有实质区别。

## 13. 当前建议

现阶段不应直接实现一个叫 “RL-assisted LeWM” 的完整系统。优先级应为：

1. 复现 LeWM 并固定 Stable World Model 评测协议；
2. 复现 RLDP、TD-JEPA、matched anti-collapse baseline，以及至少一个
   GHM/TD-Flow/UHM/Jumpy 长期模型；
3. 做 frozen-LeWM horizon diagnostic，检验长期 operator 是否提供独立信息；
4. 只有诊断成立，再设计最小 adapter objective；
5. 方法名、完整架构和论文主张留到 P0 实验之后。

当前最准确的定位不是“用 RL 防止 LeWM 坍塌”，而是：

> 研究 reward-free visual world model 中，primitive local dynamics 与 policy-conditioned
> temporally abstract dynamics 的互补性、校准方法和可归因评测。

即便这个定位也只是待检验问题。它的价值取决于是否超越 2025-2026 年已经出现的长期
world-model 与 plan-aware JEPA 工作。

## 14. 参考文献

### 领域综述与基础

- A Definition and Roadmap for World Models：<https://arxiv.org/abs/2607.06401>
- A Comprehensive Survey on World Models for Embodied AI：<https://arxiv.org/abs/2510.16732>
- World Model for Robot Learning: A Comprehensive Survey：<https://arxiv.org/abs/2605.00080>
- Video Generation Models in Robotics：<https://arxiv.org/abs/2601.07823>
- Dyna：<https://www.derongliu.org/adp/adp-cdrom/refs/sutton19900216.pdf>
- Plan2Explore：<https://proceedings.mlr.press/v119/sekar20a.html>
- General Agents Need World Models：<https://proceedings.mlr.press/v267/richens25a.html>
- World Models：<https://arxiv.org/abs/1803.10122>
- PILCO：<https://mlanthology.org/icml/2011/deisenroth2011icml-pilco/>
- PlaNet：<https://proceedings.mlr.press/v97/hafner19a.html>
- Dreamer：<https://arxiv.org/abs/1912.01603>
- DreamerV2：<https://arxiv.org/abs/2010.02193>
- DreamerV3：<https://www.nature.com/articles/s41586-025-08744-2>
- Dreamer 4：<https://arxiv.org/abs/2509.24527>
- DreamerPro：<https://proceedings.mlr.press/v162/deng22a.html>
- MuDreamer：<https://arxiv.org/abs/2405.15083>
- R2-Dreamer：<https://arxiv.org/abs/2603.18202>
- Dreamer-CDP：<https://arxiv.org/abs/2603.07083>
- DayDreamer：<https://proceedings.mlr.press/v205/wu23c.html>
- Imagination-Augmented Agents：<https://arxiv.org/abs/1707.06203>
- SimPLe：<https://arxiv.org/abs/1903.00374>
- SLAC：<https://arxiv.org/abs/1907.00953>
- MuZero：<https://www.nature.com/articles/s41586-020-03051-4>
- PETS：<https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html>
- MBPO：<https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html>
- IRIS：<https://arxiv.org/abs/2209.00588>
- TWM：<https://arxiv.org/abs/2303.07109>
- STORM：<https://arxiv.org/abs/2310.09615>
- TD-MPC：<https://arxiv.org/abs/2203.04955>
- TD-MPC2：<https://arxiv.org/abs/2310.16828>
- WIMLE：<https://arxiv.org/abs/2602.14351>

### Offline、decision-aware 与表示

- MOPO：<https://proceedings.neurips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html>
- MOReL：<https://proceedings.neurips.cc/paper/2020/hash/f7efa4f864ae9b88d43527f4b14f750f-Abstract.html>
- COMBO：<https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html>
- RAMBO：<https://proceedings.neurips.cc/paper_files/paper/2022/hash/6691c5e4a199b72dffd9c90acb63bcd6-Abstract-Conference.html>
- Objective Mismatch：<https://proceedings.mlr.press/v120/lambert20a.html>
- VAML：<https://proceedings.mlr.press/v54/farahmand17a.html>
- IterVAML：<https://papers.nips.cc/paper_files/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html>
- PAML：<https://arxiv.org/abs/2003.00030>
- Minimax Model Learning：<https://proceedings.mlr.press/v130/voloshin21a.html>
- VaGraM：<https://arxiv.org/abs/2204.01464>
- TOM：<https://proceedings.mlr.press/v211/ma23a.html>
- Value Equivalence：<https://arxiv.org/abs/2011.03506>
- Calibrated VAML：<https://proceedings.mlr.press/v267/voelcker25a.html>
- Predictron：<https://proceedings.mlr.press/v70/silver17a.html>
- Model-Based Value Expansion：<https://arxiv.org/abs/1803.00101>
- STEVE：<https://arxiv.org/abs/1807.01675>
- DeepMDP：<https://proceedings.mlr.press/v97/gelada19a.html>
- DBC：<https://arxiv.org/abs/2006.10742>
- MICo：<https://proceedings.neurips.cc/paper_files/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html>
- PSE：<https://openreview.net/forum?id=qda7-sVg84>
- BS-MPC：<https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea0206fdf3afc2ff0578a230816a9e15-Abstract-Conference.html>
- C-SWM：<https://openreview.net/forum?id=H1gax6VtDB>
- Structured World Belief：<https://arxiv.org/abs/2107.08577>
- Graph Networks as Learnable Physics Engines：<https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html>
- Predictive Representations of State：<https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html>
- Learning Causal State Representations：<https://arxiv.org/abs/1906.10437>
- Causal World Models：<https://arxiv.org/abs/2012.14228>
- Koopman Q-learning：<https://proceedings.mlr.press/v162/weissenbacher22a.html>
- Koopman-Assisted RL：<https://arxiv.org/abs/2403.02290>
- Dyn-O：<https://arxiv.org/abs/2507.03298>
- FIOC-WM：<https://arxiv.org/abs/2511.02225>

### Successor 与长期 world model

- Successor Features for Transfer：<https://arxiv.org/abs/1606.05312>
- SF Connect Model-Free and Model-Based RL：<https://www.jmlr.org/papers/v21/19-060.html>
- Forward-Backward Representation：<https://arxiv.org/abs/2103.07945>
- HILP：<https://arxiv.org/abs/2402.15567>
- FRE：<https://proceedings.mlr.press/v235/frans24a.html>
- TD-JEPA：<https://arxiv.org/abs/2510.00739>
- One-step FB：<https://arxiv.org/abs/2602.11399>
- RLDP：<https://arxiv.org/abs/2603.15857>
- `gamma`-Models：<https://proceedings.neurips.cc/paper/2020/hash/12ffb0968f2f56e51a59a6beb37b2859-Abstract.html>
- Temporal Difference Flows：<https://arxiv.org/abs/2503.09817>
- Universal Horizon Models：<https://arxiv.org/abs/2605.15603>
- Jumpy World Models：<https://arxiv.org/abs/2602.19634>
- Temporally Abstract World Models：<https://proceedings.mlr.press/v202/freed23a.html>
- World Model as a Graph：<https://proceedings.mlr.press/v139/zhang21x.html>

### JEPA、LeWM 与视觉规划

- V-JEPA：<https://arxiv.org/abs/2404.08471>
- LeJEPA / SIGReg：<https://arxiv.org/abs/2511.08544>
- SPR：<https://arxiv.org/abs/2007.05929>
- Self-Predictive RL：<https://arxiv.org/abs/2401.08898>
- DINO-WM：<https://arxiv.org/abs/2411.04983>
- PLDM：<https://arxiv.org/abs/2502.14819>
- LeWM：<https://arxiv.org/abs/2603.19312>
- V-JEPA 2：<https://arxiv.org/abs/2506.09985>
- Value-guided JEPA：<https://arxiv.org/abs/2601.00844>
- Reward-free bisimulation JEPA：<https://arxiv.org/abs/2602.18639>
- Causal-JEPA：<https://arxiv.org/abs/2602.11389>
- Temporal Straightening：<https://arxiv.org/abs/2603.12231>
- Hierarchical Planning with Latent World Models：<https://arxiv.org/abs/2604.03208>
- RC-aux：<https://arxiv.org/abs/2605.07278>
- Fast-LeWM：<https://arxiv.org/abs/2606.26217>
- AdaJEPA：<https://arxiv.org/abs/2606.32026>
- Hi-LeWM：<https://arxiv.org/abs/2607.12547>
- Temporal-Distance JEPA：<https://arxiv.org/abs/2607.25337>
- Temporally Centered SIGReg：<https://arxiv.org/abs/2607.26924>
- PhyLatent：<https://arxiv.org/abs/2608.05720>
- PSG-JEPA：<https://arxiv.org/abs/2608.06799>
- Metric Non-Collapse：<https://arxiv.org/abs/2608.07265>
- Generalization Theory for JEPA World Models：<https://arxiv.org/abs/2606.27014>

### Foundation/video world model 与 RL post-training

- UniSim：<https://arxiv.org/abs/2310.06114>
- Genie：<https://arxiv.org/abs/2402.15391>
- iVideoGPT：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/7dbb5bfab324e3b86af9bd0df15498dd-Abstract-Conference.html>
- DIAMOND：<https://arxiv.org/abs/2405.12399>
- GameNGen：<https://gamengen.github.io/>
- Cosmos：<https://arxiv.org/abs/2501.03575>
- RLVR-World：<https://proceedings.neurips.cc/paper_files/paper/2025/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html>
- RLIR：<https://arxiv.org/abs/2509.23958>
- WorldCompass：<https://arxiv.org/abs/2602.09022>
- Persistent Robot World Models：<https://arxiv.org/abs/2603.25685>
- Finetuning Offline World Models in the Real World：<https://proceedings.mlr.press/v229/feng23a.html>
- Augmented World Models：<https://proceedings.mlr.press/v139/ball21a.html>
- ReDRAW：<https://proceedings.mlr.press/v331/lanier26a.html>
- SPREAD：<https://doi.org/10.1109/LRA.2026.3688061>
- Policy-Driven World Model Adaptation：<https://arxiv.org/abs/2505.13709>

### World Action Model、机器人与显式空间模型

- Video Prediction Policy：<https://proceedings.mlr.press/v267/hu25g.html>
- DreamGen：<https://proceedings.mlr.press/v305/jang25a.html>
- VeoRL：<https://arxiv.org/abs/2505.06482>
- World Action Models are Zero-shot Policies / DreamZero：<https://arxiv.org/abs/2602.15922>
- LaWAM：<https://arxiv.org/abs/2606.15768>
- Learning Latent Action World Models In The Wild：<https://arxiv.org/abs/2601.05230>
- VLA-MBPO：<https://arxiv.org/abs/2603.20607>
- HD-VPD：<https://proceedings.mlr.press/v270/whitney25a.html>
- ParticleFormer：<https://openreview.net/forum?id=7wGYX11BJB>
- PIN-WM：<https://arxiv.org/abs/2504.16693>
- Physical Object Understanding with a Physically Controllable World Model：<https://openaccess.thecvf.com/content/CVPR2026/html/Venkatesh_Physical_Object_Understanding_with_a_Physically_Controllable_World_Model_CVPR_2026_paper.html>
- MILE：<https://arxiv.org/abs/2210.07729>
- DriveWorld：<https://openaccess.thecvf.com/content/CVPR2024/html/Min_DriveWorld_4D_Pre-trained_Scene_Understanding_via_World_Models_for_Autonomous_Driving_CVPR_2024_paper.html>
- GAIA-1：<https://arxiv.org/abs/2309.17080>
- DriveDreamer：<https://arxiv.org/abs/2309.09777>
- Latent-WAM for Autonomous Driving：<https://arxiv.org/abs/2603.24581>

### World-model evaluation 与失效诊断

- How Should World Models Be Evaluated for Embodied Decision-Making?：<https://arxiv.org/abs/2606.15032>
- WorldModelBench：<https://arxiv.org/abs/2502.20694>
- WorldBench：<https://arxiv.org/abs/2601.21282>
- 4DWorldBench：<https://arxiv.org/abs/2511.19836>
- WorldGym / Evaluating Robot Policies in a World Model：<https://arxiv.org/abs/2506.00613>
- WorldEval：<https://arxiv.org/abs/2505.19017>
- Hallucination in World Models is Predictable and Preventable：<https://arxiv.org/abs/2606.27326>
- An Optimal Tightness Bound for the Simulation Lemma：<https://rlj.cs.umass.edu/2024/papers/Paper106.html>

### 项目平台

- Stable World Model 平台：<https://galilai-group.github.io/stable-worldmodel/>
- Stable World Model baselines：<https://galilai-group.github.io/stable-worldmodel/baselines/>

# World Model 领域调研与 RL 辅助 LeWM 研究方案

调研截止：2026-08-12<br>
文档状态：领域检索完成，两批核心论文逐篇方法审计完成，方法尚未确定，实验尚未开始<br>
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
- 当前共收录 144 个 primary/official references。2026 年工作大量仍是 preprint，文中将其
  用于 novelty audit，不把作者报告的结果当作本项目已复现事实。截止日期后的新论文仍需
  在实验开始前再做一次公式级检索。
- 第 14 节已按统一模板完成 64 篇核心论文的方法审计，其中 53 篇标为全文方法审计 `F`，
  11 篇标为主体方法审计 `M`；其余 80 篇参考文献仍是待审语料库，不冒充全文已读。
- 第 14 节现在让每篇论文的完整阅读笔记只保留一个条目，并在其中单独写明发表状态、研究故事、具体方法、
  证据边界和与 TDWM 的关系。最接近本项目的工作中，TD-Flow 已正式发表于 ICML 2025，
  Jumpy、UHM、Causal-JEPA 和 Temporal Straightening 已被 ICML 2026 接收；HWM 仍在 CoRL
  2026 审稿，LeWM、Fast-LeWM、RC-aux 与 SD-JEPA 目前仍按预印本处理。

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
| TD-JEPA | 是 | 可 | 否 | 是 | TD latent prediction | direct zero-shot policy |
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

## 14. 逐篇论文阅读笔记

### 14.1 阅读口径、状态与当前范围

本节不再把“出现在参考文献中”等同于“已经读过”。每篇论文使用以下状态：

| 状态 | 判定标准 | 可用于什么结论 |
| --- | --- | --- |
| `F` | 核过全文主体、关键公式、训练与推理路径、主要实验和必要附录 | 可用于方法边界与强新颖性判断 |
| `M` | 核过完整方法链和主要实验，但尚未逐项审计全部证明或附录 | 可用于方法比较，不单独承担优先权结论 |
| `A` | 只核过摘要或官方介绍 | 只能列入候选，不用于排除创新 |
| `Q` | 已进入语料库，尚待逐篇审计 | 不得写成已读结论 |

“全文阅读”在这里不是逐字复述，而是必须回答七个可复核问题：输入数据是什么，是否使用
reward；latent 与 action 如何表示；预测算子是什么；损失和 stop-gradient 的路径是什么；测试时
模型用于 MPC、policy 还是 value；主要证据是什么；结论在哪些假设或评测范围外不成立。

两批共完成 64 篇审计。第一批覆盖 35 篇与 LeWM、RL 辅助 world model、长期模型和
world-action model 最直接相关的工作；第二批补齐 29 篇 model-based RL 基础方法、
decision-aware model learning、behavioral representation 与 LeWM 最新直接后续。其中 53 篇为
`F`，11 篇为 `M`。第 15 节其余 80 篇仍是待审语料库，不因被引用而自动变成 `F`。后续阅读
继续按“扩展领域、评测与失效诊断”的顺序写回本节，不再创建分散的调研文件。

### 14.2 阅读方式

从这里开始，每篇论文只保留一个独立条目，不再先给总表、后给方法表、最后再给故事表。
每个条目连续说明五件事：论文是否正式发表，作者想解决的旧问题，核心做法，实验真正支持的
结论，以及它与 TDWM 的关系。英文只保留无法替代的论文简称、方法名和公式变量，其余叙述用
中文。正式主会、期刊、Workshop、在审稿件和 arXiv 预印本严格区分；没有查到正式接收证据时，
统一写成“预印本”，不根据作者或机构声望推定已经发表。

### 14.3 LeWM 主干与直接前身

#### LeJEPA / SIGReg（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式主会或期刊接收记录。

**问题与模型。** LeJEPA 是通用多视图自监督方法，不是 action-conditioned world model。
encoder 预测同一样本其他 view 的均值或全局 embedding，并直接正则表示分布。

**目标与梯度。** SIGReg 对 batch 中 latent 的随机一维投影计算 Epps-Pulley 正态性统计，使
投影接近标准高斯。随机方向反复采样，从而近似约束各方向的 isotropy。方法不依赖 teacher、
EMA、stop-gradient、negative pair 或 prototype。

**它解决和没有解决的事。** 它为 LeWM 提供了无需不对称 target network 的全局非坍塌机制，
也给出特定 probe family 下 isotropic Gaussian 与下游风险的联系。但全局分布正常不意味着
action 可辨识、latent Markov、反事实分支分离或物理状态充分。LeWM 把 view prediction 换成
时序 action-conditioned prediction，并不能继承这些更强性质。

#### DINO-WM（`F`）

**发表状态。** [ICML 2025 正式发表](https://proceedings.mlr.press/v267/zhou25t.html)，已收入 PMLR 267。

**输入与模型。** 输入是 reward-free 图像动作轨迹。DINOv2 encoder 全程冻结，保留 patch
tokens；decoder-only ViT 接收 latent history 和 action embedding，以 causal attention 预测整张
future patch grid。

**目标与梯度。** 训练采用 teacher-forced multi-step latent MSE，梯度只更新 action encoder 和
predictor，不进入 DINOv2。可选 pixel decoder 独立从 feature 重建图像；论文发现让重建梯度进入
transition predictor 反而变差。

**推理与证据。** 测试时对候选 primitive action sequence 做 latent rollout，以 terminal DINO
distance 为 cost，用 CEM 或 gradient planning，实验中 CEM 更稳。它避免 end-to-end collapse
依靠的是外部预训练和冻结，不是学到了最小控制状态。约 19M 的 predictor 之外，规划时仍要运行
较大的 DINO encoder，计算口径不能只报 trainable 参数。

#### PLDM（`F`）

**发表状态。** [NeurIPS 2025 主会正式发表](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html)。

**输入与模型。** PLDM 从 reward-free offline trajectory 端到端训练 encoder，并用多个
autoregressive predictor 组成 ensemble。不同环境使用不同 encoder 和 latent 结构：Two-Room
使用 GRU/Impala 风格像素模型，Diverse PointMaze 显式拆出 spatial latent 与 velocity plane，
Ant 使用 state MLP。因此它不是一个统一像素架构在所有任务上的结果。

**精确目标。** 总目标包含五类而不是“七项”：multi-step similarity、variance、covariance、
temporal similarity/smoothness、inverse dynamics。variance/covariance 消融会造成灾难性坍塌；
temporal 项影响较温和；IDM 的帮助依赖环境。VCReg、时序约束和 IDM 共同塑造 latent，不能把
结果归因于多步预测一项。

**推理与边界。** MPPI 在 ensemble dynamics 中 rollout，cost 由 goal distance 与 ensemble
uncertainty penalty 组成，并反复 replanning。它已覆盖“reward-free、多步预测、IDM、控制导向
latent”组合，但没有 policy-conditioned successor operator，也不处理 LeWM 的 SIGReg 几何。

#### LeWorldModel（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**数据与架构。** LeWM 使用无 reward 的 `(o_{1:T},a_{1:T})` 轨迹。ViT-Tiny 约 5M 参数，
patch 14、12 层、hidden 192，以 `[CLS]` 为 state；projector 是带 BatchNorm 的单层 MLP。
predictor 是约 10M 的 causal Transformer，6 层、16 heads，动作通过每层 zero-init AdaLN 注入，
并使用长度为 `N` 的 history。

**精确目标。** teacher-forced 一步目标为

```math
\mathcal L_{\mathrm{pred}}
=\|\hat z_{t+1}-z_{t+1}\|_2^2,
\qquad
\mathcal L=\mathcal L_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{SIGReg}},
```

其中 encoder 与 predictor 均接收梯度，没有 EMA 或 stop-gradient；SIGReg 使用 1024 个随机
单位投影，默认权重约 0.1。论文还指出 encoder 最后的 LayerNorm 会与 isotropic Gaussian 目标
冲突。

**推理。** 从当前 latent 开始把 predictor 输出反复喂回自身，形成 autoregressive rollout；
terminal Euclidean distance 到 goal latent 是 cost，CEM 优化动作，执行前 `K` 个动作后重规划。

**证据与限制。** 约 15M transition、单 GPU 的训练显示 compact end-to-end JEPA 可稳定工作。
probe 和 surprise correlation 表明物理变量可读，但不是 causal sufficiency 证明。Two-Room 失败
被作者联系到真实低维流形与 isotropic Gaussian 不匹配。一步 teacher forcing 与开放环部署存在
exposure gap；deterministic point prediction 无法表达多模态未来；latent Euclidean cost 也未被
训练成 reachability 或 value。

### 14.4 直接修改 LeWM 表征、rollout 或 planner 的工作

#### Value-Guided Action Planning with JEPA（`F`）

**发表状态。** arXiv 预印本，并在 World Modeling Workshop 2026 展示；不能写成主会正式发表。

**方法。** 该工作定义 `V(s,g)=-||E(s)-E(g)||`，对无 reward trajectory 自行构造 reaching
cost `-1[s!=g]`，用 expectile IQL TD loss 学 goal-conditioned value geometry；quasimetric 版本允许
方向不对称。随后仍用 JEPA predictor 与 MPPI MPC。

**关键结果。** Wall 和 Maze 上 value/quasimetric 训练优于普通 contrastive、VCReg 或 EMA JEPA；
但把 prediction loss 与 value loss共同训练反而低于单独 value 表征。作者明确承认远距离 pair
稀疏、discount 使远处梯度变弱、随机系统下 IQL 有偏。它已覆盖“用 RL value loss 改 LeWM
planning geometry”，所以这个宽泛方案不再新颖。

#### Reward-Free Bisimulation JEPA（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 冻结 DINOv2/SimDINOv2/iBOT，在每个 patch 上加 residual MLP，把 384 维压到 32 维；
同时学 causal ViT transition。pairwise latent distance 拟合无 reward、on-policy 的一步 predicted
transition distance，并加入 PCA-based VCReg：先短暂使用普通 VICReg，再识别 pretrained feature
的主成分，只对 tail directions 保持 variance floor，使 background-dominated leading components
可以收缩。

**推理与边界。** CEM 用 terminal latent distance 做 image-goal MPC。PointMaze 的 test-time
background/distractor shift 上优于 DINO-WM，但主要证据只有简单导航、50 个 episode，且所谓
control relevance 来自行为分布上的 transition equivalence，不是全动作 bisimulation。它覆盖
reward-free dynamics-aware invariance，却不提供 policy-conditioned 长期算子。

#### Temporal Straightening（`F`）

**发表状态。** [ICML 2026 正式接收](https://agenticlearning.ai/temporal-straightening/)。

**方法。** 在 JEPA prediction 上加入连续 latent velocity 的曲率损失：

```math
v_t=z_{t+1}-z_t,
\qquad
\mathcal L_{\mathrm{curv}}
=1-\frac{v_t^\top v_{t+1}}{\|v_t\|\|v_{t+1}\|},
\qquad
\mathcal L=\mathcal L_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{curv}}.
```

prediction target stop-gradient；encoder、action encoder 和 predictor 联合更新。线性分析把
`A` 接近 identity 与有限时域 controllability Gramian 及 planning Hessian condition number 联系
起来，但非线性结论仍需控制 state-dependent Jacobian products。

**证据与边界。** Wall、UMaze、Medium Maze、PushT 上同时改善 GD 与 CEM，三 seed；长时仍有
明显 rollout drift。它已覆盖“让 LeWM latent 更适合优化”的几何正则，但 straightness 不保证
action branch 可辨识、全局不折叠或长期 occupancy 正确。

#### RC-aux（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** RC-aux 不换 LeWM backbone，而把一步 teacher forcing 换成开放环 multi-horizon loss：

```math
\mathcal L_{\mathrm{mh}}
=\sum_{k=1}^{K}w_k\|\hat z_{t+k}-z_{t+k}\|_2^2.
```

另学有方向且带预算的 `R(z,z',h)`。同轨迹 offset `Delta` 生成正例与 temporal hard negative，
跨轨迹 pair 和 stop-gradient predicted rollout pair 补充 BCE supervision。总损失为 multi-horizon
prediction、原 SIGReg 和 reachability BCE。

**推理与边界。** planner 可只用改进后的 model，也可让 reachability 对 terminal goal cost 做
soft feasibility gate。Wall 上训练目标本身把成功率从 50.4% 提到 72.4%，planner gate 再到
83.6%；额外参数约 3.74%。但负例只表示“在观察到的 trajectory 中未于预算内到达”，不是 MDP
全局不可达证明。它已直接覆盖“多 horizon + reachability 来帮助 LeWM”。

#### Fast-LeWM（`F`）

**发表状态。** arXiv 预印本，作者项目页的 BibTeX 仍为 `@misc`。

**方法。** 它不做 autoregressive state rollout，而让 causal action Transformer 把每个 prefix
`a_{t:t+k-1}` 编成 token `p_{t,k}`，再从同一 observed anchor `z_t` 并行预测所有 `z_{t+k}`：

```math
\hat z_{t+k}=G(z_t,p_{t,k}),
\qquad
\mathcal L_{\mathrm{prefix}}
=\frac1H\sum_{k=1}^{H}\|\hat z_{t+k}-z_{t+k}\|_2^2.
```

保留 SIGReg。规划可额外惩罚“直接预测终点”和“经中间 prefix 分解预测终点”的差异。

**证据与边界。** 在 LeWM 四任务同协议上平均 85.8% 到 90.5%，加 self-consistency 到 92.0%；
dynamics module 约 3.9 倍加速，完整 CEM 时间约减 48%。这已经覆盖 arbitrary-action prefix 的
多时域直接预测与 decomposition consistency。任何新方法若只做“长期头预测多步 latent”，会
与它高度重叠。

#### AdaJEPA（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 每个 MPC 周期执行首个 action chunk，把新 transition 放入 recent-N 或 hard-N buffer，
用与预训练相同的一步 latent prediction loss更新少量 encoder/predictor layers，然后重新规划。
默认 target stop-gradient，一次 gradient step，episode 结束后恢复同一 pretrained checkpoint。

**证据与边界。** PushT/PushObj 的 shape 与 visual shift、PointMaze 的 mass/damping/layout shift
均有改善，三 data seeds、每 seed 50 episodes；红色物体与 anchor 混淆时收益有限。它是
self-supervised online system identification，不是 RL reward，也不学习 policy-conditioned long
horizon operator。“执行后校准再 replanning”已经不是空白。

#### Hi-LeWM（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 冻结 LeWM encoder 与 low-level predictor。Transformer 把长度不定的 primitive action
chunk 编成 macro-action `ell`，高层 predictor 学 `p_hi(z_t,ell)->z_{t+k}`；高层 CEM 搜 latent
macro-actions 得 subgoal，低层 LeWM CEM 追 subgoal。

**失败诊断。** 无约束高层 CEM 会产生训练分布外 macro-action 和不可控 subgoal。方法最终用
训练轨迹编码出的 empirical macro bank 作为 anchor，只搜索局部 residual；staged execution 在
中等 horizon 有效，最长 horizon 又因不能在线纠错而下降。PushT 中距离比 flat LeWM 高 11.3
点，最长距离高 14.7 点，三 seed。

**边界。** 这篇论文的价值不只是 hierarchy baseline，而是说明局部与抽象模型的主要矛盾可能
在 action/support compatibility，不一定在 representation collapse。任何双层 LeWM 必须把
operator consistency 和 planner support 分开测。

#### Temporal-Distance-JEPA（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 保留 LeWM 与 SIGReg，增加 metric-residual network：对称 feature distance 加最大
ReLU asymmetric residual，得到 directed `d_psi(z_s,z_g)`。同轨迹 pair 回归 `j-i`，跨轨迹
goal 用 margin hinge；另加入与 planner horizon 一致的五步 rollout loss。完整目标为一步
prediction、rollout、temporal-distance 和 SIGReg。

**推理与证据。** topology task 直接用 `d_psi` 给 CEM/iCEM 排序；contact-rich manipulation
反而继续用同一 checkpoint 的 latent L2。锁定协议下相对 LeWM 与 RC-aux 有收益；但 PushT 上
纯 temporal cost 为 69.0%，latent L2 为 86.0%。trajectory length 只是 observed-path upper
bound，不是 shortest path。它已经覆盖 reward-free temporal progress mining、directed cost 与
train-plan rollout alignment。

#### Temporally Centered SIGReg / TC-LeWM（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 论文发现 full marginal SIGReg 会压缩多任务 mixture component centers，即便全局分布
没有坍塌也会造成 task/state aliasing。TC-LeWM 计算局部时间窗均值和 residual：

```math
\bar z_t=|W_t|^{-1}\sum_{s\in W_t}z_s,
\qquad r_t=z_t-\bar z_t,
```

只对 `r_t` 做 SIGReg，prediction 不变。若时间窗内 latent 全常数，则 residual 为零点质量，
仍会受到 Epps-Pulley penalty。

**证据与边界。** LIBERO 四 suite 的 10-task 均值 53.2% 到 73.6%，统一 40-task 为 44.4% 到
73.5%，下游是冻结 encoder 后的 task-conditioned flow-matching BC policy。它直接推翻“只要
isotropic non-collapse 就适合多任务”的假设，但不证明 residual Gaussianization 是所有任务的
最优几何。

#### PhyLatent（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 在 LeWM/SIGReg 外加入五类训练期约束：同一物理状态的颜色亮度增强 invariance；
encoded/predicted latent 到 simulator physical state 的 grounding；predicted 与 target future 的
projection/action-query alignment；batch-permuted noisy counterfactual actions 的 branch margin；
以及以 predicted future 为条件的 latent denoising。所有辅助头部署时丢弃。

**关键区分。** 它定义三种非全局坍塌：appearance 变化超过真实状态变化，物理远状态在 latent
局部碰撞，不同 action 的真实 future 分离但 predicted branches 被压缩。这比简单 variance 指标
更接近本项目原先说的“物理有意义”。

**证据与边界。** Cube 上三类 failure 分别从 15.60/6.71/8.41% 降至 7.53/0.95/4.62%，MPC
70.0% 到 78.1%；TwoRooms 提升明显，但 PushT 反而从 77.67% 降至 75.33%，三 seed。它依赖
simulator-derived physical state 和人工 augmentation，不能作为纯 reward-free self-supervision，
却是“物理 grounding”必须比较的监督上限。

#### PSG-JEPA（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 保留 LeWM teacher-forced prediction 与 SIGReg。训练期 state head 从单 latent 回归
joint、gripper、end-effector pose；transition head 从所有 horizon endpoint pair `(z_t,z_{t+k})`
回归固定维度的 joint-angle change，而不是回归多解且随 horizon 变长的 action sequence。两头
仅用于训练。

**证据与边界。** 相同 backbone 下，Cube 的五 epoch GC-IDM success 为 95.0%，LeWM 为
80.7%；30-step open-loop latent MSE 降低；LIBERO-Goal policy 为 85.3% 对 77.7%；三项真机任务
平均 79.3% 对 60.0%。它证明 privileged proprioception 能显著帮助 LeWM，但这不是 RL 帮助，
也不能支持“仅由视觉和动作自动发现全部物理变量”。

#### Metric Non-Collapse（`M`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 对 deterministic observable control system，除 prediction residual 外引入两项 state
metric hinge。local 项要求 encoder Jacobian 沿单位 tangent direction 的长度至少为 `kappa`；
global 项要求物理距离超过 `rho` 的 state pair 在 latent 中至少相隔 `alpha`。这分别阻止微分层面
坍塌和全局 folding。

**理论链。** 在 regular observable factor、Ahlfors coverage、统一 `C^{1,1}` budget、有限容量
spline approximation 等假设下，经验目标的近似 minimizer 可推出 pointwise co-Lipschitz、统一
controlled semiconjugacy，再推出 deterministic finite-horizon cost 与 optimizer transfer bound。

**边界。** 监督需要 observable state distance 和 tangent direction，不是纯 pixels-only；实验是
受控 pendulum/spline 或 bounded MLP，不能直接外推到 ViT LeWM。该文已经建立“global variance
不能排除 folding”的精确反例，所以新工作不能只用 covariance/effective-rank 声称 metric fidelity。

### 14.5 RL 梯度怎样帮助表示与无解码器世界模型

#### SPR（`F`）

**发表状态。** ICLR 2021 正式发表。

**方法。** SPR 是 Rainbow 上的在线 Atari 辅助任务。online encoder 经 action-conditioned conv
transition 连续预测 `K=5` 个 future latent；future frame 由 EMA target encoder 独立编码，online
projector/predictor 与 target projection 做归一化 cosine loss。总损失为 Rainbow Q loss 加 SPR，
两者都更新 online encoder。

**边界。** transition model 不用于 planning，论文明确把 model-based planning 留作未来工作。
无 target/stop-grad 时性能下降，作者推测并行 RL gradient 也能抑制 collapse，但没有证明 RL
单独足够。它是“RL objective 帮 self-predictive representation”的直接先例，不是 reward-free
zero-shot world model。

#### Self-Predictive Representations for RL（`F`）

**发表状态。** NeurIPS 2024 正式发表。

**理论对象。** 论文区分 observation-predictive abstraction、self-predictive abstraction、
`Q*` abstraction 与 policy abstraction。严格 self-predictive state 需要 reward prediction 与 latent
transition distribution prediction；只做 transition prediction 存在 constant solution。

**实践算法。** 在 DDPG/R2D2 类 model-free RL 上加
`||g(f(h),a)-f_target(h')||^2`，target 可 online、detach 或 EMA；不学习 reward model，不用 rollout
planning。确定系统中 L2 对理想 expected-latent prediction 合理；随机系统有 double-sampling 和
bias 问题。线性分析支持 stop-gradient/EMA 保持 feature covariance，但不是一般非线性联合收敛
保证。

**与本项目的关系。** `ZP + Q*` 在假设下可导出 reward representation，说明 RL value 与
self-prediction 互补；实验只覆盖 state MuJoCo、distractor 和 symbolic MiniGrid，论文明确不含
pixels。它不能证明给 LeWM 加 actor-critic loss会产生可用于任意 action MPC 的世界模型。

#### DreamerPro（`F`）

**发表状态。** ICML 2022 正式发表，已收入 PMLR 162。

**方法。** 保留 DreamerV2 RSSM、reward 和 imagination actor-critic，删除 observation decoder。
两条 temporally consistent augmentation sequence 的 observation features 与 RSSM states 都映射到
`K` 个 prototype；momentum encoder 产生 target assignment，Sinkhorn-Knopp 使 batch assignment
平衡，SwAV cross-view 与 temporal prediction 共同替代 reconstruction。

**边界。** anti-collapse 来自 EMA 和 balanced prototypes，reward loss在 distraction 设置还被高权重
使用。六个 DMC task、三 seed 显示 distraction robustness，但不构成“RL 本身防坍塌”的证据，
也没有 reward-free transfer 或 MPC simulator 评测。

#### MuDreamer（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式 venue。

**方法。** DreamerV3 RSSM 不再让 reconstruction gradient塑造 state。world-model heads 预测
reward、continuation、value 和 previous action；inverse dynamics 从当前 observation feature 与
前一 state 预测 action。总 world loss包含这些 prediction、dynamics KL 与 representation KL，
BatchNorm 对防坍塌关键；actor-critic 仍在 imagination 中训练。

**意义。** 这是“task/value/action heads 取代 pixels decoder”的强先例，并直接把 value target
用于 world representation。但它依赖当前任务 reward 和在线数据，不能在新 reward 上 zero-shot，
其 state 也不是 LeWM 式 image-goal MPC metric。

#### R2-Dreamer（`F`）

**发表状态。** ICLR 2026 正式发表。

**方法。** 在 DreamerV3 RSSM 中删除 decoder，用线性 projector 把 state 映到 `k`，与 detach 的
image embedding `e` 做 Barlow Twins cross-correlation：对角逼近 1，非对角逼近 0。reward、
continue 和 KL 仍保留，actor/critic 与 DreamerV3 相同。

**证据与边界。** DMC、MetaWorld 和 DMC-Subtle 上五 seed，并报告约 1.59 倍训练加速。它证明
redundancy-reduction 可以与 imagination RL 共存，不证明 actor/critic gradient 是非坍塌来源；
更不是 reward-free LeWM。

#### RLDP（`F`）

**发表状态。** ICLR 2026 正式发表。

**阶段一。** 在 reward-free state trajectory 上 unroll latent dynamics `H` 步，对 slow target
encoder 做 MSE。作者发现 target network 仍会缓慢坍塌，于是把 latent 和 prediction 归一到半径
`sqrt(d)` 的 hypersphere，并对随机 state pairs 加 orthogonality loss。

**阶段二。** 冻结 representation，训练 Forward-Backward/contrastive successor-measure model 与
task-conditioned policy；测试时用少量 reward labels 线性回归 task vector，直接执行对应 policy。

**边界。** ExoRL、SMPL humanoid、D4RL 都是 state observation，没有 pixel encoder、image-goal
MPC 或任意 action-sequence query。但它几乎正面覆盖“regularized latent dynamics 防坍塌，然后
服务 reward-free zero-shot RL”，必须是首要 baseline，而不是只在 related work 一笔带过。

### 14.6 Stable World Model 中的决策基线

#### TD-MPC2（`F`）

**发表状态。** ICLR 2024 正式发表。

**模型。** TD-MPC2 是 online model-based RL，不是 reward-free预训练。encoder `h`、latent
dynamics `d`、reward head `R`、五个 Q heads 和 stochastic policy prior `p` 共同组成 implicit
decoder-free world model；multi-task 版本还给所有模块输入 task embedding。SimNorm 把 latent
分组投到 simplex，是数值稳定手段，不是物理可辨识保证。

**梯度路径。** 对 replay sequence unroll，world-model loss 包含 predicted latent 对 stop-gradient
next observation latent 的 consistency、two-hot reward cross-entropy 和 EMA Q target 的 value
cross-entropy。actor 在 imagined latent 上最大化 Q 加 entropy，但论文明确 actor objective 只更新
`p`，不通过 actor loss 更新 encoder/dynamics。因此“TD-MPC2 用 actor gradient 塑造 world model”
是不准确的；world model 得到的是 consistency、reward 和 Q supervision。

**推理与边界。** MPPI 对 latent rollout 的累计 reward 加 terminal Q 排序，actor 提供部分候选和
warm start。原论文覆盖 104 个 online continuous-control task、三 seed，并扩展到 317M/80-task
agent。Stable World Model 的统一接口还加入 Q-ensemble uncertainty terminal penalty。它是有
reward、task-specific decision model 的强上限，不是 LeWM 的 zero-shot reward-free公平替代；
比较时必须同时报告 reward labels、online interaction 和 planner compute。

#### GCBC / GCSL（`F`）

**发表状态。** ICLR 2020 正式发表；这里把它作为 goal-conditioned imitation baseline 阅读。

**BC 的含义。** BC 是 Behavioral Cloning，即把 observation 到 action 当监督学习。GCBC 再把
goal 加入条件，最常见目标是 `||pi(o_t,g)-a_t||^2`。hindsight relabeling 把轨迹后来实际到达的
state 当 goal，于是任何 trajectory segment 都成为“朝该 future goal 的示范”。

**论文与平台实现的区别。** 原 GCSL 论文迭代执行当前 policy、把到达的 goal relabel 后重新做
supervised imitation，并在 deterministic/full-support 等假设下给出 goal-reaching lower bound。
Stable World Model 中的 GCBC 是固定离线数据上的 policy baseline，使用 DINOv2 observation/goal
features，不迭代建模环境，也不学习 value。它衡量数据中的直接模仿有多强，完全不能说明
world-model fidelity。

#### GCIQL / IQL（`F`）

**发表状态。** IQL 于 ICLR 2022 正式发表；GCIQL 是其 goal-conditioned 用法。

**方法。** IQL 只在 dataset actions 上训练 Q，避免训练阶段查询 OOD action。`V(s,g)` 对
`Q(s,a,g)` 做 upper expectile regression；Q 回归 `r(s,g)+gamma V_target(s',g)`；最后用
advantage-weighted regression 克隆高优势 action。goal-conditioned 实现通过 future/random goal
relabeling构造 sparse reaching reward。

**边界。** IQL 在 D4RL 尤其 AntMaze 展示 trajectory stitching，但它是 model-free offline RL，
没有 action-conditioned transition predictor。Stable World Model 版本使用 DINO feature，再训练
goal Q/V 与 policy。它是“给定同一数据，TD stitching 能否胜过 latent MPC”的 decision baseline，
不能作为世界模型更准确的证据。

#### GCIVL / HIQL（`F`）

**发表状态。** HIQL 于 NeurIPS 2023 正式发表。

**GCIVL。** action-free value `V(s,g)` 直接对
`r(s,g)+gamma V_target(s',g)` 做 expectile regression，不学 Q；policy extraction 仍用一步
advantage `r+gamma V(s',g)-V(s,g)` 做 AWR。去掉 Q 减少 OOD action query，也意味着 value 只
描述行为数据诱导的 state transition。

**HIQL。** 同一 action-free value 同时训练 high-level subgoal policy 与 low-level action policy，
把长目标拆成较近 waypoint。论文的价值在于 value 对远目标有噪声时，局部层次分解更稳，并能
利用 action-free data。Stable World Model 的 flat GCIVL 不等同完整 HIQL hierarchy，实验报告
时不能混称。

#### HILP（`F`）

**发表状态。** ICML 2024 正式发表。

**阶段一。** 用 goal-conditioned IVL 学 Hilbert representation，使
`V(s,g)=-||phi(s)-phi(g)||` 近似 temporal distance。作者明确承认 discount、非对称 dynamics 与
Hilbert symmetric metric 会带来 approximation error。

**阶段二。** 定义方向任务的 intrinsic reward
`<phi(s')-phi(s),z>`，对单位方向 `z` 用 offline RL 训练 latent-conditioned foundation policy。
新 reward 或 goal 可通过少量 prompt/线性方向选择快速适配。

**与 LeWM 的边界。** HILP 直接覆盖 reward-free temporal geometry 与 long-horizon policy family，
但不提供 primitive arbitrary-action dynamics 或视觉 MPC。它与 TD-JEPA/RLDP 同属 direct policy
路线，是检验“长期 operator 是否真的比普通 temporal metric 多信息”的必要 baseline。

### 14.7 时序差分、后继表示与长期世界模型

#### `gamma`-Models（`F`）

**发表状态。** NeurIPS 2020 正式发表。

**对象与 Bellman 结构。** 对固定 policy 学 normalized discounted occupancy

```math
\mu^\pi(s\mid s_t,a_t)
=(1-\gamma)\sum_{\Delta t\ge1}\gamma^{\Delta t-1}
P(s_{t+\Delta t}=s\mid s_t,a_t,\pi).
```

target distribution 是 `(1-gamma)` 的真实一步 successor 与 `gamma` 的下一状态 bootstrap
`gamma`-model 混合。论文用 GAN/f-divergence 或 normalizing flow 做 off-policy generative TD，
从而不显式标注 future timestamp 就学习长期分布。

**用途与限制。** value 可由 successor sample 的 reward期望得到，也可做 generalized rollout 和
gamma-MVE。模型是 policy-conditioned state distribution，不接收测试时任意 action sequence；
单样本 stochastic density TD 存在 bias/bound，且早期生成模型稳定性有限。

#### Temporal Difference Flows（`F`）

**发表状态。** [ICML 2025 正式发表](https://proceedings.mlr.press/v267/farebrother25a.html)并收入
PMLR 267；这是长期生成式 TD 路线中最稳固的直接前作。

**方法。** 用 flow matching 表示 policy-conditioned geometric-horizon successor measure。
普通 TD-CFM 抽样 immediate 或 bootstrap terminal sample；coupled 版本复用 noise 降 variance；
TD2-CFM 进一步对整条 probability path/vector field 写 Bellman equation，让 online vector field
匹配一步 CFM velocity 或 target vector field。

**证据与边界。** 理论给出 convergence 与较低 gradient variance；state benchmark 上改善生成
metric、value 和 GPI，但 flow sampling需要多步积分。它没有视觉 encoder，也没有 primitive
action counterfactual rollout。若新方法使用“TD 训练长期 latent distribution”，必须把它当直接
算法前身。

#### TD-JEPA（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 从 offline reward-free transition 学 state encoder `phi`、task encoder `psi`、policy
family `pi_z` 和 predictor。Monte Carlo 版本预测 successor measure 采样的 discounted future；TD
版本使用不对称 bootstrap：

```math
\|T_\phi(\phi(s),a,z)-\operatorname{sg}(\psi(s'))
-\gamma\operatorname{sg}(T^-_\phi(\phi^-(s'),a',z))\|_2^2,
\quad a'\sim\pi_z(s').
```

另有交换 encoder 角色的 symmetric path、EMA target 和 orthonormal regularizer；actor 最大化
predicted feature 与 task vector `z` 的内积。

**测试与边界。** 新 reward 由少量 labelled state 线性拟合成 `z`，直接用 `pi_z`，不是 MPC。
13 dataset、65 task 支持广泛 transfer，但理论依赖理想 tabular/linear、uniform coverage 与
orthonormal 等假设。它是“RL/TD 帮助长期 self-supervised WM”的最近工作，却没有 LeWM 的
policy-independent primitive action local simulator。

#### Universal Horizon Models（`F`）

**发表状态。** [ICML 2026 正式接收](https://rllab-snu.github.io/projects/UHM/)，作者项目页与
ICML 官方清单均有记录。

**方法。** 对 policy `pi` 学任意整数 `n` 的 `P(s_n|s_0=s,a_0=a,pi)`。`n+1` target 由真实一步
`s'` 加 target UHM 的 `n`-step sample bootstrap，flow matching 拟合 conditional distribution；
EMA target、lambda horizon schedule、winsorized geometric horizon 和 behavior-policy mixing 改善
稳定性。

**用途与边界。** UHM sample 被用于 model-based generalized TD target，reward model、critic 和
TD3+BC actor 仍需 reward。OGBench 100 task 平均优于强 baseline，但 high-dimensional humanoid
maze 失败；作者把 visual observation 与 action chunks 列为未来工作。它是任意 horizon 强 null，
不是 LeWM encoder 的防坍塌方法。

#### Jumpy World Models（`F`）

**发表状态。** ICML 2026 主会 Poster，已接收；[ICML 官方清单](https://icml.cc/Downloads/2026)
收录题目，此前也在 ICLR 2026 World Models Workshop 展示。

**方法。** 从相同 reward-free offline state 数据先训练一族 goal/foundation policies，再用
TD-Flow 学 policy 和 discount-conditioned geometric-horizon models。horizon consistency 推导
不同 discount successor measure 的 Bellman-like relation；规划时把 policy 视为 jumpy action，
每个 policy 持续随机时长，组合 GHM sample 估计终点或 subgoal value。

**证据与边界。** OGBench antmaze/cube、五类 base policy、三 seed，长 horizon 上显著优于直接
primitive ActionPlan。consistency 明显改善 EMD，但在测试 horizon 上只带来约 5% planning 增益。
论文明确把 learned visual latent 留为未来工作。它已经覆盖“跨时间尺度 consistency + 抽象动作
规划”，因此候选创新必须比这一表述更具体。

### 14.8 世界动作模型与潜动作

#### DreamZero（`M`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**方法。** 14B WAM 从 Wan video diffusion 初始化，条件为视觉历史、语言和 proprioception，
联合生成 future video latent 与 action chunk。因子化上等价于 video prediction 乘 inverse
dynamics；训练用 flow matching 对当前 noisy video/action chunk 去噪，clean previous chunks 通过
teacher forcing 提供上下文。推理时联合去噪，真实 observation 替换 cache 中预测 frame，并以
异步闭环达到约 7Hz。

**边界。** 约 500 小时 heterogeneous robot data 和 web-video prior，不使用 RL，也不是 zero-shot
reward transfer。它的 physical prior 主要来自规模化视频预训练与 joint action generation。除非
本项目转向直接 action policy，否则它是相邻 WAM，而不是 LeWM planner 的等价 baseline。

#### LaWAM（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**阶段一。** 冻结 distilled DINOv3 ViT-B/16。inverse encoder 从当前与 chunk-horizon visual
features 推断 Gaussian latent action `z`；forward decoder `LaWM(u,z)` 回归 horizon feature。另用
EEF state 与 `z` 预测 horizon state，KL 把 latent action regularize 到标准高斯。该 auxiliary head
训练后丢弃。

**阶段二。** policy prior 从当前 observation 与 language 蒸馏 posterior latent action；LaWM 单次
forward 生成 visual subgoal；Alternate-DiT action expert 在 semantic stream 与 `(u,u_hat_T)`
dynamics stream 间交替，用 conditional flow matching 生成 action chunk。Knowledge Insulation
阻止 action-expert gradient 覆盖 pretrained LaWM；不同数据用固定 physical time 而不是固定 frame
offset 对齐。

**证据与边界。** 约 3000 小时 robot video 加 1500 小时 egocentric human video，LIBERO 98.6%、
RoboTwin clean/random 92.64/89.80%、三项真机平均 90%，并显著低于 pixel WAM latency。它学习
的是 task/language-conditioned latent-action prior 与 chunk subgoal policy，不是候选 primitive
action 的 simulator，也不使用 reward-free MPC。若 TDWM 想引入 latent macro-action，它与
Hi-LeWM、LaWAM 一起构成直接先例。

### 14.9 两篇理论工作的可用结论

#### A Generalization Theory for JEPA-Based World Models（`M`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

该文把 action-conditioned spectral JEPA risk 写成 conditioned co-occurrence matrix 的低秩分解，
再把 approximation error 表示为截断后的 singular-value tail，把 sample error 表示为 encoder 与
predictor class 的 Rademacher complexity。确定 dynamics 假设下，multi-step planning error bound
随 horizon `T` 线性放大。

它提供“latent dimension 越大，approximation error 越小但 sample complexity 越差”的理论折中，
但离散 spectral setting、bounded function class 和 deterministic transition 假设不能直接证明
LeWM 的 SIGReg ViT generalization，也不涉及 policy-conditioned successor operator。

#### Metric Non-Collapse 与 JEPA Generalization Theory 的区别

前者从 state-metric hinge 推到 pointwise co-Lipschitz、uniform semiconjugacy 和 optimizer transfer，
要求更强的 state metric、smoothness 与 coverage；后者从 spectral prediction risk 推到 average
planning error，主要刻画 low-rank approximation 与 finite-sample trade-off。二者都表明不能从
训练 MSE 或 global variance 直接跳到“物理世界模型”结论，但所需监督和保证强度完全不同。

### 14.10 LeWM 最新直接邻居

#### Causal-JEPA: Learning World Models through Object-Level Latent Masking（`F`）

**发表状态。** [ICML 2026 正式接收](https://arxiv.org/abs/2602.11389)；arXiv 页面明确标注 accepted。

**表示与输入。** 每帧先经冻结的 VideoSAUR 或 SAVi object-centric encoder 变成固定数量的
slots；主要配置中的 VideoSAUR 又聚合冻结 DINOv2 features。动作和 proprioception 被编码为
独立 auxiliary entity tokens，而不是混进视觉 slot。训练时选中若干对象，把它们在整个 history
中的 slot 都换成 mask token，只保留最早时刻的 identity anchor；所有 future slots 一律被遮蔽。

**预测器与损失。** 双向 ViT 同时补全 history 中被遮蔽的对象轨迹并预测 future object slots：

```math
\mathcal L_{mask}=\mathbb E\!\left[\sum_{\tau,i}
\mathbf 1[\bar z_\tau^i\ne z_\tau^i]\|f(\bar Z)_\tau^i-z_\tau^i\|_2^2\right]
=\mathcal L_{history}+\mathcal L_{future}.
```

测试时 history 完全可见，只把未来 token mask 掉，以 Hungarian matching 后的 object cost 做
CEM。Push-T 中性能接近 DINO-WM，同时只用约 1% latent input features，规划快于其 8 倍；
CLEVRER 的 counterfactual VQA 也显著改善。

**严格边界。** 这里的“causal”是通过控制 predictor observability 迫使 interaction-dependent
prediction 的 inductive bias，不是从 observation 识别 structural causal graph；论文明确不主张
causal identifiability。它依赖 object encoder 的 slot 稳定性、identity anchor 与 finite history。
因此它直接封住“加对象级 mask 就得到因果 LeWM”的宽泛创新，却没有解决 reward/TD 联训、
长期 occupancy 或任意 behavior-policy 外动作的可识别性。

#### Hierarchical Planning with Latent World Models / HWM（`F`）

**发表状态。** arXiv 预印本；[作者主页](https://kevinghst.github.io/)标注正在 CoRL 2026 审稿，
不能写成已经接收。

**两层算子。** 低层模型 `F_1(z_t,a_t)` 预测一步 latent；action encoder `A_psi` 把一段 primitive
action chunk 压成 macro-action `l_t`；高层模型 `F_2(z_t,l_t)` 直接预测 chunk 末端 waypoint
latent。两层共享同一 observation latent 空间，高层以真实 chunk 及真实 waypoint 做
teacher-forced `L1` 预测，而不是通过 reward、skill discovery 或 policy gradient 学 abstraction。

**规划。** 高层先用 MPPI/CEM 搜 macro-actions 到最终 goal，预测序列成为 latent subgoals；低层
MPC 再搜索 primitive actions 到第一个 subgoal，执行短段后重规划。论文把同一 hierarchy 分别
接到 V-JEPA2-AC、DINO-WM 和 PLDM，表明贡献主要是 planner/model interface，而不是某一个
encoder loss。在论文的长时设置中，Franka pick-place 从 `0%` 到 `70%`，Push-T 从 `17%` 到
`61%`，最大 maze 从 `63%` 到 `83%`，并报告最多约 3 倍 planning compute 降低。

**严格边界。** macro-action decoder 没有保证任意 `l` 都对应数据支持内、低层可达的 primitive
轨迹；高层预测准确也不等于每个 waypoint 可控。它已经覆盖“给 latent WM 加高低层模型来做
长时规划”，但不是用 RL 改善 LeWM，也没有学习 policy-conditioned occupancy。

#### Subspace-Decomposed JEPAs / SD-JEPA（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式接收记录。

**核心分解。** encoder 输出被固定地切成 `k` 维 progression subspace 与 `D-k` 维 content
subspace，默认 `k=2`，实验另扫 `k∈{2,4,8}`。完整 latent 仍做 action-conditioned
next-latent prediction；SIGReg 只作用于 content，
progression 使用 temporal cosine-margin triplet，并在完整 latent 上加入 temporal straightening。
progression 的前两维写成极坐标角度 `theta=atan2(z_2,z_1)`，全部 `k` 维给出半径 `r`；predictor
也显式接收 `theta,r`，避免把时间推进全部塞进高维语义坐标。

**规划 cost。** terminal goal cost 是 content MSE、角度 cosine distance 与半径差的加权和，仍由
相同 CEM 优化；不是新 policy 或新 reward。固定、互斥坐标使两类 loss 对 latent coordinates
的直接梯度正交，但它们到共享 encoder parameters 后仍会相加，不能排除 encoder-level conflict，
更不证明两个子空间在统计上可识别为“内容”和“真实物理进度”。

**证据与边界。** 在匹配参数量、10 epochs、3 seeds 下，报告在 Reacher、Two-Room、Push-T
改善，在 Cube 略降，且最优 progression 权重依环境变化。它直接覆盖“给 LeWM 一个显式时间
进度子空间”的提议；没有 RL、successor distribution 或长时 policy occupancy。

### 14.11 经典基于模型强化学习与想象训练

#### World Models（`F`）

**发表状态。** 2018 年 arXiv 预印本；影响广泛，但不应误写成 NeurIPS 主会论文。

**方法链。** VAE 独立把 frame 压成 `z_t`；MDN-RNN 以 `(z_t,a_t,h_t)` 输出下一 latent 的
Gaussian-mixture 参数及 recurrent state；极小线性 controller 以 `[z_t,h_t]` 产生动作，并用
CMA-ES 的 episode return 优化。CarRacing 的 controller 在真实环境评估，VizDoom 还展示了在
learned dream 中训练后迁移回真环境。

**RL 到底帮了什么。** 初始 world model 用 random-policy observation 监督训练，不接收 controller
gradient。return 优化的是 controller，而非 VAE/RNN representation；文中只是提出迭代收集更难
轨迹来更新模型。MDN sampling temperature 用来缓和 controller 利用模型漏洞。

**边界。** pixel reconstruction、随机数据覆盖和小任务使它与 LeWM 相距很远。它是“policy 在
world model 中训练”的先例，不是“RL loss 防止 world-model collapse”的证据。

#### PILCO（`F`）

**发表状态。** ICML 2011 正式发表。

**模型与不确定性传播。** 对低维 state difference 学独立 Gaussian Processes：
`x_{t+1}=x_t+f(x_t,u_t)+epsilon`。当前 state distribution 和确定 policy 共同形成不确定 GP 输入，
再用 analytic moment matching 把每一步预测近似为 Gaussian，递推到整个 horizon。

**policy update。** 目标是累积期望 cost `J(theta)=sum_t E[c(x_t)]`。由于 GP predictive moments、
moment matching 和 policy 都可微，PILCO 用解析链式法则计算 `dJ/dtheta`，再由共轭梯度或
L-BFGS 优化 policy；每次真实 trial 后把 transition 加回数据并重拟合 GP。

**边界。** RL 的 cost 训练 policy，模型仍按 probabilistic transition likelihood 学；不确定性让
少样本 planning 更稳，但 GP 复杂度、Gaussian propagation 和低维 state 假设限制了视觉扩展。

#### PETS（`F`）

**发表状态。** NeurIPS 2018 正式发表。

**模型。** 多个 probabilistic neural networks 用 Gaussian NLL 预测 state delta 的均值和方差；
bootstrap ensemble 表示 epistemic uncertainty，单网络的输出方差表示 aleatoric uncertainty。
trajectory sampling 用 particles 在 ensemble 中传播，`TS1` 每步重采样模型，`TS∞`
让同一 particle 整段固定在一个模型上，避免把模型分歧错误平均掉。

**控制闭环。** CEM 迭代拟合高回报 action-sequence distribution，按已知 reward 在 particles 上
估计候选回报，只执行第一步，再收集真实 transition 更新 ensemble。RL/return 没有反向塑造
representation；它决定采哪些数据和如何查询模型。

**边界。** 原实验是低维 state、短 MPC、已知 reward。PETS 支持“RL 主动改善模型覆盖”，但不
支持“value loss 让视觉 latent 更物理”。

#### MBPO（`F`）

**发表状态。** NeurIPS 2019 正式发表。

**方法。** 用真实 replay 拟合 probabilistic ensemble dynamics；从 replay 中真实 state 随机分叉，
每个分叉只滚动 `k` 个 model steps，把 synthetic transitions 放入单独 model buffer，再用真实与
模型数据共同训练 SAC。`k` 随训练日程增长，但核心结果是短 rollout，甚至一步 rollout，通常
比长 rollout 更可靠。

**关键结论。** prediction accuracy 不是 policy utility 的充分条件：论文观察到即使可做约 200 步
准确 rollout，policy 用长 synthetic rollout 仍可能更差。MBPO 通过大量短分叉扩大数据，而不是
要求单条 imagination 很长。

**边界。** RL 主要消费模型样本；标准 maximum-likelihood model loss 不由 SAC value 加权。
因此它是 LeWM 后接 policy learning 的 baseline 设计，不是 joint representation objective。

#### Plan2Explore（`F`）

**发表状态。** ICML 2020 正式发表，已收入 PMLR 119。

**world model 与内在奖励。** 先用 Dreamer RSSM 学 observation/reward-free latent dynamics，另训
一个 one-step latent ensemble。对同一 imagined next state 各 predictor 的输出方差作为
disagreement，近似模型信息增益。

**探索。** actor-critic 完全在 imagination 中最大化未来 disagreement，而不只是当前 novelty；
执行探索 policy 收集真实数据后更新 RSSM。下游 task 到来时冻结已学的 task-agnostic model，
用少量 reward labels 训练 reward predictor，再在 imagination 中训练 task policy。

**边界。** 这是 RL 帮 world model 最清楚的一条路径：RL 改变真实数据分布，而不是直接反传到
encoder。它需要 online interaction，不能原样用于固定 offline LeWM dataset。

#### PlaNet（`M`）

**发表状态。** ICML 2019 正式发表，已收入 PMLR 97。

**方法链。** Recurrent State-Space Model 同时保留 deterministic recurrent state 与 stochastic
state；observation encoder 给 posterior，transition prior 从上一 latent/action 预测，decoder 和
reward head 用变分序列目标训练。latent overshooting 让多步 prior 与未来 posterior 对齐，降低
只做一步 teacher forcing 的 drift。

**推理。** 测试时不用显式 policy，而在 latent model 中用 CEM 搜 action sequence、累计预测
reward，只执行第一步。模型同时有 observation 和 reward likelihood，故不是 reward-free JEPA。
本轮已核主体方法与实验，尚未逐项复查全部附录超参数，因此保守标 `M`。

#### Dreamer（`F`）

**发表状态。** ICLR 2020 正式发表。

**模型。** 以真实 replay 训练 RSSM 的 representation/posterior、transition/prior、observation 与
reward heads；latent KL 约束 posterior 可被 dynamics 预测。随后从真实序列的 posterior states
出发，在纯 latent imagination 中展开当前 actor。

**RL update。** critic 拟合 imagined lambda-return；actor 最大化这些回报，并通过 reparameterized
latent dynamics 与 reward/value predictions 反传 analytic gradients。相比 CEM，actor amortizes
planning，部署时一次前向出动作。

**边界。** actor/value 学习依赖模型，但 model objective 仍是 replay 上的变分预测目标；不能把
“actor gradient 穿过模型求动作梯度”误写成“actor loss 更新了 world model”。

#### DreamerV2（`F`）

**发表状态。** ICLR 2021 正式发表。

**相对 Dreamer 的改变。** stochastic latent 改为多组 categorical variables，并以
straight-through estimator 训练；world-model loss 含 image、reward、discount likelihood 与 KL。
KL balancing 对 prior 与 posterior 两侧使用不同 stop-gradient 权重，使 dynamics 追 posterior
比 representation 追 prior 更快。

**actor-critic。** actor 在 imagined trajectories 上结合 REINFORCE 与 straight-through dynamics
gradient，critic 拟合 lambda-return。论文实现明确把 world model 与 behavior learning 分开：
actor/value gradients 不更新 world-model parameters。

**边界。** categorical latent 和 KL balancing 是稳定 RSSM 的技术，不提供 reward-free physical
state 保证；Atari return 也不能证明 arbitrary-goal revaluation。

#### DreamerV3（`F`）

**发表状态。** 2023 年先公开预印本，后在 Nature 2025 正式发表。

**稳健统一化。** 仍用 categorical RSSM，但把 dynamics KL 与 representation KL 用 stop-gradient
拆开并加 free bits；对大幅度 reward/value 用 symlog 和 two-hot regression，按 return percentile
范围归一化 advantage，并用 continuation head 处理 episode termination。这些设计使同一超参数
覆盖 150 多个任务。

**训练边界。** world model 在真实 replay 上最小化 reconstruction、reward、continue 与 KL；
actor/critic 在 imagined lambda-returns 上更新。RL 决定数据和行为模块，但 actor loss不是一个
LeWM anti-collapse regularizer。其表示是 task-aware RSSM，跨 reward 零样本规划不是主要接口。

#### Dreamer 4（`F`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式主会或期刊接收记录。

**foundation world model。** 先以大规模视频和动作预训练 causal video tokenizer 与 interactive
dynamics transformer；用 shortcut forcing/diffusion-forcing 式 `x`-prediction 同时处理历史真实
tokens 与 noisy future，使模型既能 teacher-forced 训练也能长时生成。下游再接 action、reward、
continue 等 task heads。

**policy learning。** actor/value 在视频模型生成的 imagination 中训练，默认冻结庞大的 transformer，
并引入 behavioral prior 约束动作。Minecraft diamond 结果依赖约 2B 模型与约 2500 小时 VPT
数据，回答的是 scalable video imagination 能否支撑 RL，而不是轻量 latent JEPA 是否防坍塌。

**边界。** 它是重要规模上界和 foundation baseline；将其成功简单归因为“RL 训练了 world
model”是错误的，主要 world-model pretraining 与下游 policy optimization 在梯度上分离。

#### SimPLe（`F`）

**发表状态。** ICLR 2020 正式发表。

**模型和循环。** stochastic video model 从最近 frames 与 action 预测下一 RGB frame、reward 与
done。训练使用 pixel loss、reward loss、scheduled sampling 等稳定化；agent 与真实 Atari 交互
收集新数据，重训模型，再把模型当环境运行 PPO，循环多轮。

**控制 model bias。** imagined episode 每约 50 步就从真实 replay frame 重新开始，并对截断处
用 value bootstrap，避免 PPO 长时间利用错误模型。RL 不直接修正预测 loss，而是通过访问分布
持续暴露模型缺口。

**边界。** pixel generation 和 Atari task reward 与 LeWM 的 goal-latent MPC 不同；它证明的是
小数据下 learned simulator 可供 policy training，不是 latent 满秩或物理可识别性。

#### IRIS（`F`）

**发表状态。** ICLR 2023 正式发表。

**离散 world model。** VQ autoencoder 把每帧编码成一串离散 tokens；autoregressive GPT 按
`frame tokens, action, next-frame tokens` 的序列预测下一帧 token，同时预测 reward 和 done。
解码 token 可得到 imagined frame，但 policy 直接消费 latent/token history。

**RL。** 按 DreamerV2 风格在模型生成的 trajectories 中训练 actor-critic，再到真实 Atari 收集
数据并更新 tokenizer/GPT。相比 continuous RSSM，它把多模态视觉未来交给 categorical token
language model，但每帧逐 token 自回归带来计算成本和 compounding error。

**边界。** IRIS 是 reconstruction/generation-based、online、task-reward-aware，不是 fixed-data
reward-free LeWM；它可作为“离散生成模型 + imagination RL”的架构对照。

#### MuZero（`M`）

**发表状态。** Nature 2020 正式发表。

**隐式模型接口。** representation `h` 把 observation history 变为 latent state；recurrent dynamics
`g(s,a)` 输出下一 latent 与 immediate reward；prediction `f(s)` 输出 policy logits 与 value。
MCTS 只在 latent 中调用 `g/f`，无需重建 observation。

**训练。** 从 replay state 展开真实 action sequence，对每步预测 reward、bootstrapped/n-step
value target 和 MCTS improved policy，representation、dynamics、prediction 联合更新。因此模型
只需 value-equivalent，不需在像素或“物理变量”意义上准确。

**边界。** 这是“RL targets 定义隐式 world model”的核心先例，也最能反驳“decision-useful 等于
physical”。本轮核对方法主链和主要证据，未重新审计 Nature 补充材料全部实现细节，标 `M`。

#### TD-MPC（`M`）

**发表状态。** ICLR 2022 正式发表。

**任务导向 latent。** encoder、action-conditioned latent dynamics、reward predictor、双 Q、policy
prior 共同训练。loss 把多步 latent consistency、reward regression 与 TD value prediction 合在
一条 unroll 上，EMA target 提供下一 latent/Q target。

**规划。** MPC 在短 horizon latent rollout 中累积 predicted rewards，以 terminal Q 补上长尾；
候选既可从 Gaussian sampling 也可由 learned policy proposal 提供，执行第一步后重规划。它把
local model 与 model-free value 的互补性做得很直接。

**边界。** 所有 representation 都围绕当前 task reward/Q，不能当作 reward-free、goal-agnostic
LeWM。因本轮未逐表复核全部消融，暂标 `M`；完整继任者 TD-MPC2 已在第一批标 `F`。

### 14.12 决策感知模型学习：RL 怎样直接改变模型目标

#### Value-Aware Model Learning / VAML（`F`）

**发表状态。** AISTATS 2017 正式发表，已收入 PMLR 54。

**目标。** maximum likelihood 要在所有 state directions 拟合 transition；VAML 只要求模型在
planner 会查询的 value functions 上给出相同 expectation。典型 robust 形式是：

```math
\mathbb E_{s,a}\sup_{V\in\mathcal F}
\left|\mathbb E_{s'\sim P}[V(s')]-\mathbb E_{s'\sim\hat P}[V(s')]\right|^2.
```

当 `F` 是 1-Lipschitz functions 时，它与 Wasserstein discrepancy 相连。模型可以在 pixel/state
prediction 上“不真实”，却让 Bellman backup 准确。

**边界。** 这是 RL/value 真正进入 model loss 的直接前作，但需要选择 value function class；类太
窄会丢掉 revaluation 所需信息，太宽又退回完整 dynamics。它优化 decision equivalence，不保证
物理可解释 latent 或 arbitrary downstream goals。

#### Iterative VAML（`M`）

**发表状态。** NeurIPS 2018 正式发表。

**方法。** 不再对一个巨大 value class 求最坏情况，而是在 approximate value iteration 中交替：
用当前 policy/value 采数据；让模型匹配当前 value 在真 transition 与模型 transition 下的
expectation；再用新模型更新 value/policy。模型准确性跟随实际生成的 value-function sequence。

**意义与边界。** 它把 VAML 从 robust minimax 变成可实现的 moving target，并给出有限样本分析；
但模型强烈绑定 planner 迭代路径，换 reward/value 后可能失效。实验范围和附录未在本轮完全
复核，标 `M`。

#### Policy-Aware Model Learning / PAML（`M`）

**发表状态。** arXiv 预印本；截至调研截止日未核到正式主会或期刊版本。

**目标。** PAML 不匹配 value expectation，而直接让模型环境产生的 policy gradient 接近真实
环境 gradient，例如最小化

```math
\|\nabla_\theta J_P(\pi_\theta)-\nabla_\theta J_{\hat P}(\pi_\theta)\|_2^2.
```

论文分别讨论 REINFORCE 与 deterministic policy-gradient 近似，因而 model loss 同时依赖当前
policy、visitation distribution、reward/value estimator 和 planner。

**边界。** 它是“RL gradient 教 model 哪些误差重要”的最直接先例之一，但只保证当前 policy
update 方向，不保证 model rollout、goal replacement 或 policy 外 action query。主体方法已核，
完整证明和所有实验设置未逐项审计，标 `M`。

#### Value Gradient weighted Model Learning / VaGraM（`F`）

**发表状态。** ICLR 2022 正式发表。

**推导与实现。** 对真实下一 state 邻域的一阶 Taylor expansion 表明，transition error 沿
`∇V(s')` 大的方向更影响 value。为避开 VAML 在模型 OOD state 上查询不可靠 value 和
iso-value spurious minima，VaGraM 使用稳定的 diagonal upper-bound 形式：

```math
\mathcal L_{VaGraM}\approx
\|\operatorname{diag}(\nabla_{s'}V(s'))(\hat s'-s')\|_2^2.
```

它保留 supervised next-state target，只把每个 state dimension 按 value sensitivity 加权，能嵌入
Dyna-style actor-critic。

**边界。** 一阶局部近似依赖 value gradient 的质量和坐标系；若用 learned visual latent，坐标
旋转会改变 diagonal weighting。它是 LeWM 加 value-gradient loss 必须比较的直接 baseline，
但本身不提供 reward-free universality。

#### Transition Occupancy Matching / TOM（`F`）

**发表状态。** UAI 2023 正式发表，已收入 PMLR 211。

**对象。** 定义当前 policy 下真实/模型 transition occupancy：
`d_T^pi((s,a),s')=d_T^pi(s,a)T(s'|s,a)`，并最小化两者的 f-divergence，而不是均匀拟合 replay
中的所有 transition。理论上由此给出 model-based return lower bound。

**实际算法。** 用分类器判别真实与模型 transition occupancy，再把密度比与 policy relevance
Q/importance weights 转成加权 maximum-likelihood model update；policy 在模型内更新并收集新
真实数据，形成 on-policy 循环。

**边界。** TOM 优先保证当前 policy 访问处的模型容量，不是 arbitrary-action、policy-independent
世界模型。它与候选长期 occupancy head 名称相近但方向相反：TOM 用 occupancy 加权 local
transition，TD-Flow/Jumpy 直接预测 occupancy distribution。

#### Value Equivalence（`F`）

**发表状态。** NeurIPS 2020 正式发表。

**定义。** 给定 policy 集 `Π` 与 value-function 集 `V`，若对所有 `π ∈ Π, v ∈ V` 都有
`T^πv = T_hat^πv`，两个模型在该集合上 value equivalent。扩大 policy/value 集会收紧可接受
模型类；收缩集合则允许大量与真实 dynamics 不同但足以支持指定 planning updates 的模型。

**意义。** 该框架统一解释 MuZero、Predictron 与 value-aware models 为什么可以不重建真实
observation。它给“模型充分性”一个可操作的 decision-relative 定义。

**边界。** value equivalence 不是 physical equivalence、causal identifiability 或 task-general
world model。候选方法若声称“更真实”，必须另报 rollout/controllability/OOD revaluation，不能
只用 control return 证明。

#### Calibrated Value-Aware Model Learning（`M`）

**发表状态。** ICML 2025 正式发表，已收入 PMLR 267。

**发现。** 常用 value-aware/MuZero-style surrogate 在 stochastic environments 中可能不
calibrated：即使数据无限、优化达到总体最优，也未必恢复正确 value/model。论文区分“一个
deterministic model 能直接预测某个 value”与“model distribution 对后续 planning 校准”两件事，
并构造 corrected calibrated objectives。

**对本项目的警告。** 直接加 value head 或 TD target 可能得到 task return 好、但 distribution
错误的 latent operator；一旦换 goal/reward 就暴露问题。当前核过主体定理、反例和主实验，
尚未逐证明审计全部附录，标 `M`。

### 14.13 结构化表示与行为表示

#### DeepMDP（`M`）

**发表状态。** ICML 2019 正式发表，已收入 PMLR 97。

**方法。** encoder 把 observation 映射到 latent MDP，同时学习 latent reward 与 latent transition
distribution。理论用 reward prediction error 和 transition distribution discrepancy 界定原 MDP
与 latent MDP 的 value-function 差异，并把两个预测任务作为 Atari policy 的 auxiliary losses。

**边界。** latent 保存的是当前 reward/value 所需信息，不是 reward-free 完整动力学；理论 metric
与视觉 encoder 的实际优化之间仍有 gap。本轮核主体理论与实验，未逐证明复核，标 `M`。

#### Deep Bisimulation for Control / DBC（`F`）

**发表状态。** ICLR 2021 正式发表。

**目标。** 随机取两个 transitions，使 latent distance 逼近 bisimulation metric target：即时
reward difference 加 discount 后的 latent transition Wasserstein distance。Gaussian transition
下用闭式 `W_2`，target encoder/next latent 通过 stop-gradient 稳定训练，再与 SAC 联训。

```math
\|z_i-z_j\|_1 \approx |r_i-r_j|+
\gamma W_2(\hat P(\cdot|z_i,a_i),\hat P(\cdot|z_j,a_j)).
```

**边界。** reward 相同且未来 reward behavior 相同的视觉状态应被主动合并，这是 task-aware
invariance，不是“保留全部物理因素”。因此 DBC 能防止无意义 pixel detail 主导 latent，却可能
正当地坍塌掉换任务后重要的信息。

#### MICo（`F`）

**发表状态。** NeurIPS 2021 正式发表。

**对象与更新。** MICo 避免 bisimulation 的 optimal-transport coupling，定义当前 policy 下两个
states 的 diffuse behavioral distance：

```math
U^\pi(x,y)=|r^\pi(x)-r^\pi(y)|+
\gamma\mathbb E_{x'\sim P_x^\pi,y'\sim P_y^\pi}U^\pi(x',y').
```

从 replay 独立采两个 next states 即可做 TD regression，并让 learned representation 的角度/范数
距离拟合该 target，计算比 bisimulation 简单。

**边界。** independent coupling 使它是 diffuse metric，甚至可能有非零 self-distance；它刻画
policy-conditioned behavioral similarity，不学习供 MPC 查询的 action-conditioned simulator。

#### Contrastive Learning of Structured World Models / C-SWM（`F`）

**发表状态。** ICLR 2020 正式发表。

**结构。** CNN object extractor 把 frame 分成多个 object slots；共享 object encoder 得到 states；
graph neural network 按 action 和其他对象消息预测每个 object 的 state delta。positive energy 是
预测 next slots 与真实 next slots 的平方误差，随机 negative state 通过 margin 被推远：

```math
H=\frac1K\sum_k d\!\left(z_t^k+T^k(z_t,a_t),z_{t+1}^k\right),\qquad
\widetilde H=\frac1K\sum_k d\!\left(\widetilde z_t^k,z_{t+1}^k\right),
\qquad \mathcal L=H+\max(0,\gamma-\widetilde H).
```

**用途与边界。** 训练完全 reward-free，主要以 multi-step rollout retrieval/ranking 检查对象与
关系是否被学到。它假定 object factorization 和相对简单的 action作用方式；contrastive ranking
好不等于 goal cost calibrated，也没有 LeWM 式端到端 MPC 证据。

#### Structured World Belief（`M`）

**发表状态。** ICML 2021 正式发表。

**belief 而非单 state。** 在部分可观测场景中维护多个 object-centric particles；Sequential Monte
Carlo 对 action-conditioned dynamics 推进、用 observation likelihood 加权并 resample。object file
把实体“存在”与当前“可见”分开，slot matching 保持身份，glimpse/attention 只更新观测到部分，
从而表示 object permanence 与多模态假设。

**控制与边界。** world belief 可先在 random-policy data 上训练，再供 A2C 或 planning 使用。
它解决的是 partial observability 与 uncertainty，而非 JEPA collapse；代价是 reconstruction/
likelihood、object discovery 和 particle inference 的复杂组合。本轮未逐项复核全部附录，标 `M`。

### 14.14 两批审计后的创新边界

经过两批方法审计，以下主张已经被直接前作覆盖，不能再作为 TDWM 的核心创新：

1. “RL/value loss 让 JEPA latent 更适合规划”：Value-Guided JEPA、SPR、MuDreamer 已覆盖；
2. “多步或开放环训练减少 LeWM rollout drift”：PLDM、RC-aux、Fast-LeWM 已覆盖；
3. “学 temporal distance/reachability 改进 LeWM cost”：RC-aux、Temporal-Distance-JEPA 已覆盖；
4. “跨 horizon consistency 形成长期模型”：Fast-LeWM、UHM、Jumpy 已覆盖不同接口；
5. “普通 SIGReg 不够，修正其 latent geometry”：TC-LeWM、PhyLatent、Metric Non-Collapse 已覆盖；
6. “物理 state/action grounding 让 LeWM 有物理意义”：PhyLatent、PSG-JEPA 已直接覆盖；
7. “测试时用新 transition 自监督适配”：AdaJEPA 已覆盖；
8. “加高层 latent action/subgoal 改善长时规划”：Hi-LeWM、LaWAM 已覆盖，并暴露 support gap；
9. “用 TD 学 reward-free 长期 future 再 zero-shot control”：`gamma`-model、TD-Flow、TD-JEPA、
   RLDP 和 Jumpy 已形成完整谱系。
10. “用 RL 主动收集对 world model 最有信息的数据”：PETS 与 Plan2Explore 已覆盖；
11. “用 value 或 policy gradient 只训练决策相关 dynamics”：VAML、PAML、VaGraM、TOM、
    Value Equivalence 和 Calibrated VAML 已形成理论与算法谱系；
12. “在 learned model 里训练 policy 就等于 RL 改善了模型”：World Models、MBPO、Dreamer 系列、
    SimPLe 与 IRIS 反而说明二者常在梯度上分离，不能据此立项；
13. “对象级 masking 或显式时间子空间让 LeWM 更物理”：Causal-JEPA 与 SD-JEPA 已直接覆盖；
14. “高低层 world model 加 macro-action 改善长时规划”：HWM 已在三类 latent WM 上验证。

仍可能值得研究的不是这些模块的简单相加，而是一个更窄、可证伪的接口问题：

> 在同一个 reward-free visual latent 中，能否同时保留 LeWM 对任意 primitive action prefix 的
> counterfactual local operator，以及 TD-JEPA/Jumpy 对 policy-conditioned 长期 occupancy 的
> temporally abstract operator，并用可检验的 compatibility 条件证明或实证二者提供互补信息，
> 而不是 Fast-LeWM 式多 horizon 重参数化、RC-aux 式 reachability head，或 Hi-LeWM 式 hierarchy？

这个问题也不能直接宣称新颖。最低限度必须通过以下判别：长期头在控制参数量和训练数据后，
对 frozen-LeWM 的误差是否有条件互信息；local-to-occupancy consistency 是否优于分别训练；提升
是否在同一 solver/budget 下仍存在；是否超出 behavior-policy support；以及优势是否来自新的
operator relation，而不是额外 reward、proprioception、policy capacity 或 planner compute。

### 14.15 整个领域合起来的故事，以及 TDWM 还能讲什么

整个领域的故事可以压成一句话：**研究者先尝试把世界完整地生成出来，随后发现控制只需要部分
世界；再发现过度任务化的模型不能换目标，于是回到 reward-free latent prediction；最后又发现
非坍塌、一步准确和视觉语义仍不等于长期可控、物理可信。**

因此，以下故事已经不够成立：

1. “我们首次用 RL 防止 LeWM collapse。”SPR、MuDreamer、R2-Dreamer、VAML 与 TD-MPC 系列
   已从 auxiliary prediction、task heads、value-aware loss 等不同方向覆盖。
2. “RL 让 latent 更有物理意义。”RL 通常只让 latent 对当前 reward/value 更充分，甚至会删除
   task-irrelevant physics；这和 task-general physical model 是相反拉力。
3. “长时 TD target 解决 LeWM rollout。”TD-JEPA、TD-Flow、UHM、Jumpy 已覆盖长期 target，
   但换来 policy conditioning，未自动保留任意 primitive-action counterfactual。
4. “成功率提高就证明模型更好。”planner、cost、policy prior、额外 reward 和 compute 都能提高
   success，必须用固定 solver 的 model fidelity、candidate ranking 与 OOD revaluation 拆开归因。

仍可能成立的研究故事应更窄：**LeWM 擅长 reward-free、policy-independent 的局部反事实动作
预测；successor/TD 模型擅长 policy-conditioned 的长期结果。两者各自丢失对方的信息。能否让
长期 RL operator 只作为可审计的 consistency teacher 或训练诊断，改善 local model 的长时
decision fidelity，同时用 gradient isolation、policy diversity 和 reward replacement 实验证明
没有把 LeWM 退化成 task-specific value model？**

这个故事的核心不是“再加一个 TD loss”，而是证明一个此前没有被同时满足的三角关系：

```text
任意 primitive action 的局部反事实能力
        + reward/policy 变化后的可重估性
        + policy-conditioned 长期预测的低误差
```

若实验只能改善第三项，却损害前两项，研究结论应是“RL 把 LeWM 任务化了”，而不是“RL 帮助
LeWM 学到了更好的世界模型”。这个反例本身也有研究价值，并且比预设方法一定成功更可信。

## 15. 参考文献

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

- GCSL / GCBC：<https://arxiv.org/abs/1912.06088>
- Implicit Q-Learning：<https://arxiv.org/abs/2110.06169>
- HIQL / GCIVL：<https://arxiv.org/abs/2307.11949>
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
- Subspace-Decomposed JEPAs / SD-JEPA：<https://arxiv.org/abs/2605.31111>
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

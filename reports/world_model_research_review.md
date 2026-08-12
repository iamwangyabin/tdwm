# World Model 领域调研与 RL 辅助 LeWM 研究方案

调研截止：2026-08-13<br>
文档状态：领域检索完成，64 篇核心论文全文审读完成，方法尚未确定，实验尚未开始<br>
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
- “把自监督 latent prediction、RL/decision supervision 和显式 anti-collapse 放在同一系统”已有
  SPR、Self-Predictive RL、DreamerPro、MuDreamer、R2-Dreamer、TD-JEPA 和 RLDP；但这些
  工作并未证明 RL 单独足以防坍塌；
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
- 第 14 节已按统一模板完成 64 篇核心论文的全文审读：正文、关键公式、训练与推理路径、主要
  实验以及判断边界所需附录均已核对。63 篇标为完整审计 `F`；Metric Non-Collapse 因最关键
  理论假设未在实验中满足，保留 `M`，不把“读过全文”误写成“结论已被充分验证”。其余 80 篇
  参考文献仍只是扩展语料库，不因列入 bibliography 冒充逐篇全文审读。
- 第 14 节现在让每篇论文的完整阅读笔记只保留一个条目，并在其中单独写明发表状态、研究故事、具体方法、
  证据边界和与 TDWM 的关系。最接近本项目的工作中，TD-Flow 已正式发表于 ICML 2025，
  UHM、Causal-JEPA 和 Temporal Straightening 已有 ICML 2026 正式证据；Jumpy 已进入 ICML
  2026 官方材料，但本次未单独核到最终 PMLR camera-ready 页面；HWM、LeWM、Fast-LeWM、
  RC-aux 与 SD-JEPA 目前仍按预印本处理。

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
中学习。PlaNet 用 RSSM 从像素学习 belief/dynamics，再用 CEM
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

PlaNet 论文虽然推导并实验了 latent overshooting，但最终 RSSM agent 没有采用它；附录显示该项
帮助 DRNN、却略微降低 RSSM 表现。这里不能把“论文提出过的 auxiliary loss”误写成最终系统
成功的必要组件。

Reconstruction-free MBRL 也不是 LeWM 才出现。DreamerPro 用 prototype assignment 取代
像素重建，并把 recurrent temporal state 蒸馏进 prototype；MuDreamer 以 value 和 previous
action prediction 取代 reconstruction，并用 batch normalization 防 collapse；R2-Dreamer
用 redundancy reduction 作为内部正则，在不依赖 decoder 或 data augmentation 时防止
collapse；Dreamer-CDP 则把 JEPA-style continuous deterministic prediction 接入 Dreamer。
因此，“decoder-free + RL/decision learning + 显式 anti-collapse”已经是明确竞争线，而不是待填
空白；但 RL 的独立 anti-collapse 因果作用仍未被这些方法证明。

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

本轮共完成 64 篇核心论文的全文审读，覆盖 LeWM/JEPA 直接前后续、RL 表征、successor 与长期
模型、world-action model、经典 model-based RL、decision-aware model learning 和结构化表示。
每篇都核了正文、关键公式、数据与梯度路径、主要实验，以及支撑优缺点判断所需的附录。63 篇
标为 `F`；Metric Non-Collapse 虽已全文阅读，但理论保证依赖的关键条件没有在经验设置中实现，
因此谨慎保留 `M`。第 15 节其余 80 篇仍是扩展语料库，不因被引用而自动变成“已逐篇全文审读”。
后续新增阅读继续写回本节，不创建分散的正式调研文件。

### 14.2 阅读方式

从这里开始，每篇论文只保留一个独立条目，不再先给总表、后给方法表、最后再给故事表。
每个条目连续说明五件事：论文是否正式发表，作者想解决的旧问题，核心做法，实验真正支持的
结论，以及它与 TDWM 的关系。英文只保留无法替代的论文简称、方法名和公式变量，其余叙述用
中文。正式主会、期刊、Workshop、在审稿件和 arXiv 预印本严格区分；没有查到正式接收证据时，
统一写成“预印本”，不根据作者或机构声望推定已经发表。

### 14.3 LeWM 主干与直接前身

#### LeJEPA / SIGReg（`F`）

**准确题名、作者与发表状态。** Randall Balestriero、Yann LeCun，*LeJEPA: Provable and
Scalable Self-Supervised Learning Without the Heuristics*，2025；主来源为
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)（本次核读 v3）。截至调研日未核到正式主会或
期刊 proceedings，应写“arXiv 预印本”，不能仅因作者或传播度写成已发表。

**作者真正要解决什么。** 这不是 action-conditioned world model，而是通用多视图自监督表示
学习。普通 JEPA 的 view-prediction 只要求同一对象的增强视图表示一致，常数表示同样可以使损失
最小；过去方法靠 stop-gradient、EMA teacher、prototype、negative 或手写 variance/covariance
约束避免坍塌。LeJEPA 试图把这堆经验装置换成一个明确分布目标：在固定总体方差等条件下，
isotropic Gaussian 对论文分析的线性 probe 和局部非线性 estimator family 有较好条件数和有限
样本风险（Sec. 3，Theorem 1）。这里的“最优”受 probe/task family 和正则性假设限制，不是
“高斯 latent 自动等于物理状态”。

**模型、数据流和梯度。** 一个样本生成多个 augmented views，全部经共享 backbone/projector；
global views 还经过 predictor，再去预测其他 view 或 global-view embedding 的均值。所有分支
默认都可反向传播，没有 target encoder、EMA 或 stop-gradient；SWA 只是可选训练增强
（Sec. 5.1-5.2，Algorithms 1-2）。因此预测项提供 invariance，SIGReg 单独承担全局 non-collapse。

**SIGReg 到底做了什么。** 对每个 minibatch 重采单位球方向 `u_j`，把高维 embedding 投影为
一维样本 `Zu_j`，再用 Epps-Pulley 特征函数统计量度量它与 `N(0,1)` 的差异，最后在方向上平均
（Definition 2，Algorithm 1）。Cramer-Wold 方向判别把“所有一维投影为标准高斯”连接到联合
isotropic Gaussian（Lemma 3，Theorem 2）；只匹配有限阶矩不够唯一确定分布（Theorem 3）。
Epps-Pulley 的 loss、gradient 和 curvature 有界（Theorem 4），minibatch gradient bias 随样本量
消失（Theorem 6）。实现用 17 个 quadrature knots，建议区间 `[-5,5]`，默认 1024 个方向；方向
逐步重采，因此单步只用 16 个也会在训练过程中累积覆盖（Sec. 4.2-4.3）。总目标是 view
prediction 与每个 view 的 SIGReg 加权和，推荐 `lambda` 约 0.05。

**实验真正支持什么。** 论文覆盖 10 多个数据集、60 多种 backbone，规模到 1.8B ViT-g，证据
主要是识别/迁移而非控制。ImageNet-1K frozen online-linear 中 ViT-L/14 为 77.1%，
ConvNeXtV2-H 为 78.5%，摘要给 ViT-H/14 约 79%（Sec. 6，Table 6）。跨数据集 all-shot 平均，
LeJEPA ViT-L 为 79.48，I-JEPA ViT-H 为 78.50，I-JEPA+STOP 为 80.70（Table 2），所以它并非
每项都压过其他 JEPA。Galaxy10 的 in-domain 小 ResNet frozen probe 约 75%-78%，图 1 中
ResNet34 full fine-tune/frozen 为 83.28/78.17，DINOv3 frozen 为 71.38。Table 1、7 和
Figures 7-14 显示 batch 128 仍能工作，512-4096 directions 差异有限，register token 非必需，
`lambda` 有较宽稳定区间。原文一处称八个 views，另一处默认 `g=2,l=8`；这项内部不一致需要按
代码核验，不能自行改写。

**优点、缺点与 TDWM 关系。** 优点是全局防坍塌目标独立、可诊断、可扩展，且无需 teacher。
但它只约束边缘分布：一个全局标准高斯 latent 仍可把物理远状态折叠、压缩不同动作分支、保留
nuisance 或破坏 reachability；论文也没有 action、dynamics、Markov sufficiency 或规划实验。
因此 SIGReg 适合保留为 LeWM 的全局底座，RL/value/successor 信号则应补局部状态区分、方向性
可达性与控制几何。若新方法替掉 SIGReg，还必须重新验证不会整体坍塌。

#### DINO-WM（`F`）

**准确题名、作者与发表状态。** Gaoyue Zhou、Hengkai Pan、Yann LeCun、Lerrel Pinto，
*DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*；
[ICML 2025 正式论文](https://proceedings.mlr.press/v267/zhou25t.html)，PMLR 267:79115-79135；
预印本为 [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)。这篇确实已经主会发表。

**故事和问题设定。** 作者想证明：无需 reward、expert policy 或像素重建，只在离线
observation-action trajectory 上预测强视觉 encoder 的 future patch feature，也能在测试时为新
goal 搜动作。这里“zero-shot”是 world model 训练后不为每个新 goal 再训练 policy/reward，
不是没有该环境的动力学数据（Sec. 1、3）。

**数据流与模型。** RGB 经冻结 DINOv2 ViT-S/14，主模型保留 `14x14=196` 个 384-D patch
tokens；CLS-only 只是消融。可选 proprioception 被拼入 patch，动作经 MLP 到 10 维后复制到各
patch。去掉 tokenizer 的 6 层 decoder-only ViT（16 heads、MLP 2048，约 19M）用 frame-level
causal mask 读取 history 和 action，一次预测下一 frame 全部 patch，而非逐 token autoregression
（Sec. 3.1，App. A.4）。正文/附录对输入先 resize 到 224 还是 196 的叙述不完全一致，复现应查
released code。

**损失、梯度与推理。** dynamics 用 teacher-forced multi-step latent MSE；DINOv2 全程冻结，
所以动作不能反向塑造视觉表示。可选 transposed-conv image decoder 只为可视化单独训练，
decoder loss 不进入 dynamics。测试时编码 current/goal，递归 rollout candidates，以 terminal
patch-tensor MSE 为 goal cost，CEM 做 MPC/replanning；也测试直接 gradient planning，但更差
（Sec. 3.2）。因此 planner 使用的是 DINO 视觉距离，不是经过 value/reachability 学习的成本。

**数据和主结果。** 六任务 PointMaze/Wall/Reacher/PushT/Rope/Granular 的 Table 1：DINO-WM
为 `.98/.96/.92/.90/.41/.26`，后两项是越低越好的 Chamfer distance；DreamerV3 为
`1.00/1.00/.64/.30/2.49/1.05`，IRIS 为 `.74/.04/.18/.32/1.11/.37`，去 reward 的 TD-MPC2
多项为 0。数据附录给 PointMaze 2000x100、Wall 1920x50、Reacher 3000x100、PushT 18,500
条 noisy expert，Rope/Granular 各 1000 条。Rope/Granular 文字说轨迹长 20，Table 11 又列
length 5，这也是需按代码核实的内部冲突。

**关键消融。** Table 2 的 DINO-patch 是 `.98/.96/.92/.90/.41/.26`，DINO-CLS
`.96/.58/.60/.44/.84/.79`，R3M `.94/.34/.40/.42/1.13/.95`，ResNet
`.98/.12/.06/.20/1.08/.90`；成功很大部分来自保留 spatial patch，而不只是“DINO语义强”。
OOD Table 3 的 WallRandom/PushObj/GranularRandom 为 `.82/.34/.63`，Dreamer 为
`.76/.18/1.53`。PushT 数据从 200 增到 18,500 trajectories 时，成功率 `.08→.92`、LPIPS
`.056→.005`（Table 5），说明仍强依赖覆盖量。history 2 时无/有 causal mask `.36/.88`，
history 3 为 `.08/.92`（Table 6）；加 decoder loss `.80`，不加 `.92`（Table 7）。PointMaze、
PushT 的 CEM 为 `.98/.90`，open-loop `.80/.86`，gradient `.22/.28`（Table 8）。A6000 上
batch32 单次 forward `.014s`，但 `100 candidates x 10 iterations` 完整规划约 53s，不能拿前者
冒充端到端控制延迟。

**优点、局限与 TDWM 关系。** 它有扎实的 patch/encoder/mask/decoder/data-size 消融，是
reward-free latent planner 的强基线。局限是物理抽象来自外部视觉预训练，encoder 固定；terminal
DINO distance 不保证可达；确定性点预测不表示多模态，规划昂贵，测试环境变化有限。它是
“无 RL shaping”的重要对照：TDWM 必须在冻结/端到端 encoder、相同数据和 planner budget 下
证明 RL 改善的是 action-conditioned reachability 或 cost ranking，而不是借了更强预训练。

#### PLDM（`F`）

**准确题名、作者与发表状态。** Uladzislau Sobal（预印本署名 Vlad Sobal）、Wancong Zhang、
Kyunghyun Cho、Randall Balestriero、Tim G. J. Rudner、Yann LeCun，*Learning from Reward-Free
Offline Data: A Case for Planning with Latent Dynamics Models*；
[NeurIPS 2025 Main Conference Track 正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html)，预印本
[arXiv:2502.14819](https://arxiv.org/abs/2502.14819)。旧版本曾以 workshop 形式出现，但最终状态
已是 NeurIPS 主会，旧报告若只写 workshop 需要更正。

**论文真正的故事。** 核心不是仅提出一个 JEPA，而是比较 reward-free offline data 下两条路线：
goal-conditioned/model-free RL 学 policy/value，和学 latent dynamics 后 test-time optimal control。
作者在 23 个导航数据设置中改变 expert fraction、trajectory length、数据量和 layout diversity，
分析 HILP、HIQL、GCIQL、CRL、GCBC 与 PLDM 各自何时占优（Sec. 1、4）。

**模型、损失和梯度。** encoder `h` 输出 latent，多个 autoregressive predictor `f_k` 构成
ensemble。不同环境用不同结构：TwoRooms 是 ImpalaSmall+GRU512（约 2.2M）；Diverse
PointMaze 是 `16x26x26` spatial latent 加两张 velocity planes，conv predictor 仅 53,666 参数；
Ant-U-Maze 直接用去掉 `x,y` 的 29-D proprio state，经 MLP 到 256，ensemble=5（App. D）。
因此不是同一个 pixels-only 架构横扫全部任务。完整目标（App. D.1.1）为 multi-step latent
similarity `L_sim`、batch-time variance floor `L_var`、off-diagonal covariance `L_cov`、相邻 latent
temporal smoothness `L_time-sim` 和从 `(z_t,z_{t+1})` 预测动作的 `L_IDM`。VCReg防整体/低秩
坍塌，temporal项约束局部平滑，IDM保留控制信息；不能把结果只归因于 multi-step prediction。

**规划。** ensemble 在 candidate action 下递归 rollout；goal cost 汇总多个 horizon/ensemble 的
goal distance，uncertainty cost 是各 latent 维 ensemble variance 的折扣和。MPPI 最小化二者并
replan（Sec. 3.3）。所以它已有 uncertainty-aware planning，但没有 RL value shaping。

**实验中最重要的正反证据。** 高质量 TwoRooms 数据中 PLDM `97.8±0.7`，HILP `100`，GCIQL
约 `98`（Table 2）。但 no-door-passing stitching 中，PLDM 只有 `34.4±2.7`，GCIQL `99.6`、
HILP `100`；因此“model planning 天生最会 trajectory stitching”不成立。Figure 4 系统扫 expert
fraction、轨迹长度 91/64/32/16 和数据量；PLDM/GCIQL 数据效率好，HILP/GCIQL 在 stitching
更强。未见 layout 数 5/10/20/40 的实验中 PLDM 总体最好（Figs. 2、8），但图线没有可可靠抄录
的精确表值，本报告不伪造数值。Ant-U-Maze 用 5M transitions，短序列条件下 PLDM、HIQL、
HILP 均到 100%（Fig. 6）。

规划开销（Table 5）：每步 replan 16s/归一成功1.00，每4步 4.8s/.95，每16步 2.6s/.90；
GCIQL/HIQL policy 推理约3.6/4.0s。重建对照（Table 6）：PLDM97.4、rewardless DreamerV3 24.0、
reconstruction26.2、rewardless TD-MPC2 0、TD-MPC2+IDM35（后者单 seed）。组件消融（Table
D.1.2）：full 在 TwoRoom/Diverse Maze 为 `98.0±1.5/98.7±2.8`；去 variance `13.4/11.4`，去
covariance `29.2/7.8`，去 temporal smoothness `71.0/95.6`，去 IDM `98.0/75.5`。这说明全局
anti-collapse 是成败项，IDM 则环境相关。

**优点、局限与 TDWM 关系。** 优点是认真研究数据性质和 RL/control 边界，而非只报一个最好
数字。局限是导航占主导、架构/privileged velocity/raw-state 跨环境变化、deterministic dynamics、
MPPI/ensemble 较重。它给 TDWM 最有价值的启示是：RL 表示在 stitching/长期 reachability 上有
真优势，dynamics planning 在 layout OOD、数据效率和 cost 重用上有优势。新方案应把 RL 的
value/successor/occupancy 当 LeWM 的表示或 cost 辅助，而非用 policy 替代 model，并在
no-door-passing 类支持缺口上专门检验。

#### LeWorldModel（`F`）

**准确题名、作者与发表状态。** Lucas Maes、Quentin Le Lidec、Damien Scieur、Yann LeCun、
Randall Balestriero，*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
Pixels*，2026；[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)（本次核读 v3）。截至调研日只
核到预印本和项目代码，未核到正式接收。

**故事和数据。** DINO-WM靠冻结 foundation encoder，PLDM靠多项 anti-collapse/auxiliary。
LeWM 试图只用“next latent prediction + SIGReg”从 raw pixels 端到端训练 encoder/predictor，不要
重建、EMA teacher、pretraining 或 reward，并直接用 latent distance 做 goal CEM（Sec. 1、3）。
数据是离线 reward-free `(o_{1:T},a_{1:T})`，但作者明确依赖 pseudo-expert/exploratory dataset
具有足够 action/state coverage。

**模型与信息流。** 224x224 frame 经从零训练 ViT-Tiny（patch14、12层、3 heads、hidden192，
约5M），取 CLS 后过一层 MLP+BatchNorm projector。projector 不只是装饰：作者指出 ViT 最后
LayerNorm 与 Gaussian target 冲突。predictor 是 6层、16 heads、dropout .1 的 causal
Transformer（约10M），动作通过每层 zero-init AdaLN 注入，history 采用 frame-causal mask
（Sec. 3.1-3.2，App. D）。target frame 经同一个可训练 encoder，没有 stop-gradient/EMA，故
prediction 和 SIGReg 都会更新 encoder。

**目标和规划。** `L_pred=||Pred(z_history,a)-z_next||^2` 是 teacher-forced 一步 MSE；SIGReg 在
history、batch、latent 样本上用 1024 projections；总目标
`L_pred+lambda L_SIGReg`，默认约 `.09-.1`。不训练 decoder；可选 decoder 只为解释表示，而且
decoder loss 回传会伤规划。测试从 current history 递归预测，CEM 以 terminal
`||z_hat_T-z_goal||_2` 排序，执行 action block 后重规划。这个 Euclidean cost 没有受 value、
steps-to-goal 或 reachability 监督，正是 train-plan gap。

**实现细节。** frame skip5、batch128；训练 window 为4 frames，每相邻 frame 对应5 primitive
actions。PushT/Cube history3，TwoRoom history1。CEM 300 candidates/top30；PushT 30 iterations，
多数其余任务10；model horizon5覆盖25环境步。附录文字称执行完整 optimized sequence 后再
replan，具体 receding interval 应以 released config 核验（Apps. E-F）。

**结果和消融。** Figure 6 图读值约 TwoRoom/Reacher/PushT/Cube `87/86/96/74`；这些不是统一
置信区间表。PushT 三 seed 的精确 Table 5：DINO-WM `92±1.63`、PLDM `78±5.00`、LeWM
`96±2.83`。固定近似 compute 的 Figure 3 中 LeWM PushT/Cube约`90/74`，DINO约`13/48`；完整
planning `.98s` 对 `47s`，但“48x”依赖 token、候选和实现。PushT linear probe（Table 1）：agent
location MSE/r，LeWM `.052/.974`、PLDM `.090/.955`、DINO `1.888/.977`；block `.029/.986`、
`.122/.938`、`.006/.997`；angle `.187/.902`、`.446/.745`、`.050/.979`。这证明物理量可读，
也显示 DINO 在 block/angle 更线性可读；probe 不等于 planner 一定用对这些信息。VoE 中 teleport
使 prediction surprise 显著上升（多项 `p<.01`），color change较弱（Fig. 8），这不是物理定律或
因果 sufficiency 证明。

附录消融：`lambda .01-.2` 多数仍高成功，`.5`下降；latent dim约184后饱和。predictor
Tiny/Small/Base `80.67/96/86.7`；decoder loss无/有`96/86`；ViT/ResNet`96/94`；dropout
`0/.1/.2/.5` 为`78/96/85.33/66.67`。solver Table 10：CEM96、SGD26、RMSProp67.33、Adam84，
反而说明 latent geometry 尚不足以让局部梯度规划稳定。

**优点、缺点与 TDWM 关系。** 它让动作数据能塑造小型视觉 encoder，是做受控新方法的最好
底座。局限是一时步 teacher forcing、deterministic mean future、全局而非局部 SIGReg、无方向的
terminal L2、有限 coverage；作者也承认 long-rollout drift 和低维 TwoRoom 与 Gaussian target 的
不匹配。TDWM 最合理的插入点是保留 `prediction+SIGReg`，再用 offline RL 的 goal value、
successor/occupancy 或 reachability 让 planner cost/表示具有长期行为意义；必须同时监控新 loss
是否破坏 prediction、contact geometry 与 non-collapse。

### 14.4 直接修改 LeWM 表征、rollout 或 planner 的工作

#### Value-Guided Action Planning with JEPA（`F`）

**准确题名、作者与发表状态。** Matthieu Destrade、Oumayma Bounou、Quentin Le Lidec、
Jean Ponce、Yann LeCun，*Value-Guided Action Planning with JEPA World Models*；
[arXiv:2601.00844](https://arxiv.org/abs/2601.00844)。PDF 明示“Presented as a poster at the World
Modeling Workshop 2026”，准确状态是 workshop poster / arXiv 预印本，不是主会论文。

**故事。** LeWM/DINO-WM 把 latent L2 临时当 goal cost，但 prediction loss 没要求它等于
steps-to-goal；障碍两侧视觉相近、控制距离却很远。作者将缺口直接改写成 goal-conditioned
offline RL：让 embedding distance 逼近 reaching task 的负 optimal value，使 planner geometry
反映可达进度（Sec. 3）。这是本节最直接的“RL 帮 JEPA world model”工作。

**模型、数据流和损失。** 图像经约2.2M的 conv/res encoder 到512-D flat latent；action encoder
为 identity，约1.3M的 MLP predictor 接收 state/action concat，训练片段长16。定义
`V_theta(s,g)=-||E_theta(s)-E_theta(g)||^2`，每步 reaching cost 为 `1[s!=g]`。TD residual 是
`-1[s_t!=g]+gamma*sg(V(s_{t+1},g))-V(s_t,g)`，用
`|tau-1[x<0]|x^2` 做 expectile regression（Eq. 1）；goal 来自轨迹最后状态和 batch 随机状态。
quasimetric variant 允许 `d(s,g) != d(g,s)`，但正文未给其完整逐层网络，复现应查引用实现，
不能凭印象补写。`Sep` 先只用 value loss 训 encoder，再训 predictor；`Joint` 让 value 与
prediction 同时更新全网。推理仍用 predictor rollout 和 MPPI，不直接学 policy。

**实验。** Wall-Small/Big 各1000条随机轨迹、length64；Maze 1000条、length101、5个train
layouts。MPPI 在 Wall 用2000 perturbations、std12、temperature .005、horizon96/64；Maze用
500、std5、temperature .0025、horizon100。base lr .0028；VF 的`gamma/tau=.98/.80`，quasi为
`.93/.60`，由另一份 WS validation 调参（Sec. 4，App. 7.3）。Table 2 的 WS/WB/Maze：
contrastive `.49/.59/.50`，prediction+VCReg `.55/.89/.54`，prediction+EMA `.46/.43/.04`，
VF `.63/.94/.49`，VF+prediction `.55/.75/.49`，VF-quasi `.71/.96/.63`，
VF-quasi+prediction `.61/.85/.43`，Joint `.47/.67/.39`。

**如何解读。** 优点是直接学习方向性 goal cost，并认真比较分阶段/联合训练。最重要的负结果是
最好的是 **quasi value-only**，把 prediction 加回来反而下降；所以它证明 RL 能训练更好的
planner metric，却没有证明 RL 已改善 action-conditioned dynamics model。局限还有简单 wall/maze、
远距离 pair 稀疏、discount 梯度衰减、数据 support 和 stochastic IQL bias，没有 LeWM+SIGReg
与 contact-rich robot。因而“给 LeWM 加 IQL/value loss”的宽泛方案已经不新；TDWM要解决的是
value-prediction 梯度冲突，例如把 policy-conditioned progress 放入受约束 subspace/cost head，
同时保持 dynamics、SIGReg 和 contact geometry。

#### Reward-Free Bisimulation JEPA（`F`）

**准确题名、作者与发表状态。** Leonardo F. Toso、Davit Shadunts、Yunyang Lu、Nihal Sharma、
Donglin Zhan、Nam H. Nguyen、James Anderson，*Learning Invariant Visual Representations for
Planning with Joint-Embedding Predictive World Models*；
[arXiv:2602.18639](https://arxiv.org/abs/2602.18639)，当前为预印本。“Reward-Free Bisimulation
JEPA”是本文分类名，不是论文正式题名。

**故事和结构。** 冻结 DINO patch 虽有空间语义，也保留背景、灯光、moving distractor 等 slow
features。作者在 DINOv2/SimDINOv2/iBOT 后给每个384-D patch共享 residual MLP，降到32-D并
保留196个位置；latent坐标量约缩12倍。causal Transformer预测下一 compressed latent。这样希望
距离主要反映 transition similarity，而非 nuisance（Secs. 1、3）。

**目标。** `L_dyn` 对齐 `h(f(o_{t+1}))` 与 `T(h(f(o_t)),a_t)`。两条样本的 target dissimilarity
`Delta=gamma||T(w_t,a_t)-T(w'_t,a'_t)||`，再令
`L_bisim=(||w_t-w'_t||-Delta)^2`。标准 bisimulation还有 reward difference，本文为 reward-free
而删除，所以它只是行为数据上的 transition-equivalence proxy，不是完整 reward/value
bisimulation。防坍塌先用 VICReg warm-up，再一次 PCA；variance floor只施加 tail PCs，允许作者
认为由背景主导的 leading PCs 收缩，covariance项仍压全部 off-diagonal。该做法依赖 nuisance 在
leading PCs 的启发式。正文称50epochs并在50后切PCA，另一图注又提90epochs，时序需查代码。

**推理和证据。** planner仍以 terminal compressed-latent L2 用CEM。唯一主任务是 PointMaze，
2000条随机轨迹，六种 test visual conditions。Table 1 DINO-WM 为
`.80/.72/.60/.56/.48/.78`，domain randomization `.82/.82/.82/.68/.64/.82`，本文
`.78/.80/.76/.86/.78/.82`：clean 略低，多种 shift 更稳。encoder消融（Table 2）：无预训练
`.68/.44/.70/.26/.36/.64`，DINO `.78/.80/.76/.86/.78/.82`，SimDINO
`.40/.38/.36/.42/.42/.36`，iBOT `.72/.70/.74/.72/.72/.72`，故“agnostic”不代表各 encoder
同样有效。

**优点、局限与 TDWM 关系。** 优点是正面测 nuisance OOD、保留 spatial patches并有 domain
randomization 对照。局限是 transition target从学习模型自身 bootstrap，sample动作不相同也不
遍历全动作，无法推出真正 all-action bisimulation；仅一个简单maze，未测长期 occupancy、contact
或 stochasticity，且 encoder不端到端。TDWM若用RL/bisimulation删 nuisance，应在 LeWM 上学习
policy-conditioned long-horizon transition distribution，而不只是两条 observed one-step transition
的距离。

#### Temporal Straightening（`F`）

**准确题名、作者与发表状态。** Ying Wang、Oumayma Bounou、Gaoyue Zhou、Randall
Balestriero、Tim G. J. Rudner、Yann LeCun、Mengye Ren，*Temporal Straightening for Latent
Planning*；[arXiv:2603.12231](https://arxiv.org/abs/2603.12231)。作者
[官方项目页](https://agenticlearning.ai/temporal-straightening/)确认被 ICML 2026 接收，故是正式
主会论文。

**故事和模型。** 一步 dynamics 即使准确，latent trajectory若弯曲/折返，terminal L2就不是可行
geodesic proxy，gradient action optimization也会病态。视觉路径用冻结DINO+可学习projector，
或从零ResNet；action encoder和ViT predictor联合训练。prediction为
`||z_hat_{t+1}-sg(z_{t+1})||^2`，曲率为
`1-cos(z_{t+1}-z_t,z_{t+2}-z_{t+1})`，两者加权。target stop-gradient；曲率更新encoder/
projector。它不是严格anti-collapse机制，常数表示理论上仍可满足局部方向关系。

**理论边界。** 在线性 `z'=Az+Ba` 中，若 `||A-I||<=epsilon`，terminal-MSE Hessian 与有限时域
controllability Gramian相关，condition上界含
`kappa(B)^2((1+epsilon)/(1-epsilon))^(2(K-1))`（Sec. 4）。低action维只在controllable subspace
成立。训练中的轨迹cosine不等价于全空间`A≈I`，非线性 state-dependent Jacobian products也不
被该界完整控制。

**数据和结果。** Wall1920x50、UMaze2000x100、Medium4000x100、PushT18,500条；frame
skip5、history3、batch32。GD planner用Adam、lr .1、100steps，primitive horizon25；也测CEM。
spatial DINO+projector 的 no/straight（Table 1）：Wall open `.80→.9067`、MPC `.9067→1.00`；
UMaze `.44→.94`、`.8133→1.00`；Medium `.72→.8267`、`.9667→.9867`；PushT
`.70→.7733`、`.7867→.8533`。从零ResNet的PushT MPC `.7067→.9133`。但global DINO的
PushT MPC `.1133→.0867`，说明spatial representation选择比曲率还关键。

CEM Table 5：Wall `.92→1.00`、UMaze `.7533→.94`、Medium `.9267→.8667`（变差）、PushT
`.7133→.80`。长horizon50时PushT MPC `.2733→.24`也下降；Medium `.65→.88`。latent dim
8/32较好，2太小、128常变差；普通 temporal smoothness/contrastive替代没有同样收益。

**优点、局限与 TDWM 关系。** 它直接优化 planner geometry，测了GD、CEM、long horizon和
spatial/global消融。局限是局部straight不排全局folding、不表达方向/不可逆性、不保证action
branches或reachability，且若干CEM/MPC配置下降。它已占据“几何正则让JEPA更可规划”的位置；
TDWM必须展示RL successor/value提供的是全局长期ranking或support，而非曲率的换皮，并与
`L_curv`做正交消融。

#### RC-aux（`F`）

**准确题名、作者与发表状态。** Wenyuan Li、Guang Li、Keisuke Maeda、Takahiro Ogawa、
Miki Haseyama，*Predictive but Not Plannable: RC-aux for Latent World Models*；
[arXiv:2605.07278](https://arxiv.org/abs/2605.07278)，当前为预印本。

**故事与模型流。** LeWM训练一步teacher forcing，却让CEM对开放环rollout排序；latent近也不
表示在剩余budget内可达。RC-aux保留LeWM encoder/predictor/SIGReg，从真实context出发递归喂
expert actions和前一预测，令
`L_mh=sum_k w_k||z_hat_{t+k}-z_{t+k}||^2`，以开放环multi-horizon替代一步loss。它再训练有序
head `R_phi(z,z',h)`，估计目标能否在预算h内到达（Sec. 3）。

**reachability labels与梯度。** 同轨迹`i<j`，offset `Delta=j-i`，随机预算h的label为
`1[h>=Delta]`，`h<Delta`作为temporal hard negative；跨trajectory pairs作负例。还把
stop-gradient predicted intermediate latent与未来encoded target配对，使head见planner分布。
BCE有class weight和predicted-pair权重；总目标是`L_mh+alpha SIGReg+beta L_reach`。planner
基础terminal L2乘上`max(m,1-lambda_plan R)`，只给高reachability candidate打折；
`lambda_plan=0`可分离training与gate收益。

**实验。** TwoRoom/Reacher/PushT/Wall/Cube，每组50 episodes、5组。主结果 RC 对 matched
controls：`98.0 vs88.8`、`87.2 vs82.8`、`90.8 vs91.2`（略差）、`83.6 vs50.4`、`76 vs72.8`。
关planner gate后（Table 2）TwoRoom93.2、Reacher81.2、Wall72.4；打开后98/87.2/83.6，说明
Wall/TwoRoom训练项本身有用，Reacher增益主要来自gate。paired table中Wall RC-only85 vs
LeWM-only2，TwoRoom24 vs1，PushT14 vs13，Cube15 vs6。

参数18.034M→18.710M，+3.74%，一次reach call增<.8ms。LIBERO扩展不是MPC，而是frozen
encoder后action head：LeWM .712、RC .812、repeat-tuned .864、OpenVLA-7B .970；只能说明
表示对该policy协议有帮助。

**优点、局限与 TDWM 关系。** 它把horizon mismatch、reachability表示和planner gate分开测，
predicted-latent pairs也触达rollout分布。根本局限是`j-i`仅为观察路径长度，不是shortest path；
`h<Delta`和cross-trajectory negatives都可能是假负例。`lambda_plan`还按任务调（Wall .85、多个
任务.35、Cube0），PushT无增益。“multi-horizon+budget reachability帮LeWM”已被直接覆盖；
TDWM应以offline RL的policy-conditioned successor/occupancy缓解observed-path假负例，并在锁定
协议与RC-aux比较，否则只是换head/label。

#### Fast-LeWM（`F`）

**准确题名、作者与发表状态。** Yuntian Gao、Xiangyu Xu，*Fast LeWorldModel*；
[arXiv:2606.26217](https://arxiv.org/abs/2606.26217)。论文/项目 BibTeX 均为 arXiv `@misc`，当前
是预印本。

**故事和信息流。** LeWM每个candidate要串行调用H次transition，误差还会沿 predicted state
累积。Fast-LeWM把预测单元改为“从同一 observed anchor执行某 action prefix后的状态”。causal
action Transformer先读由`z_t`映射的state token，再读动作；第k个token只能看前k个动作，输出
`p_{t,k}`。共享6层action-modulated residual MLP从同一真`z_t`和所有prefix tokens并行预测
`z_hat_{t+k}`。prefix Transformer为3层、6heads、dim192；predictor hidden2048、fusion768、
AdaLN-zero、dropout.1；总17.9M，接近released LeWM18.0M（Sec. 3）。

**损失与推理。** `L_prefix=H^{-1}sum_k||z_hat_{t+k}-z_{t+k}||^2+lambda SIGReg`，每个prefix都有
target，encoder、prefix encoder、predictor联合更新。训练10epochs，batch128（Cube32），horizon
clamp `[1,5]`，history1。规划只取最后prefix terminal latent，以L2和CEM排序，一次forward完成
多horizon。可选self-consistency在plan time比较直接H-prefix终点与“经中间prefix分段”终点，
beta=1；不是主要训练loss。

**实验。** Table 1 TwoRoom/Reacher/PushT/Cube：PLDM `97/78/78/65`，DINO-WM
`100/79/74/86`，LeWM `87/86/96/74`，Fast `98/88/96/80`，+consistency
`98/90/98/82`；平均85.8→90.5→92.0。baseline多是原论文point estimate，不能都视为matched
multi-seed。RTX4090、相同CEM budget的TwoRoom：LeWM 5 calls，dynamics31.4s、total54.4s；
Fast 1 call，8.0s/28.3s，即module3.92x、完整时间约减48%（Table 2）。probe中angle linear
MSE/r `.314/.828`，反而差于LeWM `.187/.902`，虽MLP `.009/.995`更好。消融（Table 4）
long-action LeWM `76/70/80/58`，terminal-only Fast `96/80/90/72`，dense full
`98/88/96/80`，无state token `94/82/92/80`。

**边界与 TDWM。** 优点是改 predictor interface而非换encoder取巧，速度口径和dense-vs-terminal
消融清楚。局限是最长只训5个model steps，超长仍需组合；确定性direct prediction无uncertainty，
四任务且完整训练方差有限；self-consistency可能惩罚stochastic但合理的不同future。它不解决
reachability/folding。任何“长期head直接预测多步latent”已与它重叠；TDWM若加长期RL信息，
应在相同prefix backbone上检验收益来自successor/value结构还是仅dense direct prediction。

#### AdaJEPA（`F`）

**准确题名、作者与发表状态。** Ying Wang、Oumayma Bounou、Yann LeCun、Mengye Ren，
*AdaJEPA: An Adaptive Latent World Model*；[arXiv:2606.32026](https://arxiv.org/abs/2606.32026)，
当前为预印本。

**故事和算法。** 离线world model部署时通常冻结；外观、shape、mass/damping/layout变化会使
prediction失真。AdaJEPA在每个episode从同一checkpoint开始，循环 plan、执行第一个action
chunk、观察真实`o'`、adapt、replan（Algorithm 1）。recent-N或hard-N buffer存最近或最大误差
transition；最小化`mean||Pred(Enc(o),a)-sg(Enc(o'))||^2`。默认只更新predictor最后block/final
LN和encoder最后projection，一步gradient；lr分别`5e-4/1e-5`，recent buffer=5。episode后reset，
所以不是跨episode continual learning。大多数base model是Temporal Straightening的ResNet
stop-grad/curvature模型，不是LeWM+SIGReg；另有DINO-WM实验，报告中不能笼统称“在线更新
LeWM”。

**实验。** shift包括PushObj unseen shapes，PushT blur/salt-pepper/dark/color，PointMaze
mass/damping/unseen layouts；3 test seeds x50 episodes，最多20 MPC steps。PointMaze frozen
default GD/CEM `82.7/84.0`、lowmass`77.3/82.0`、damping`77.3/76.0`、layout`53.3/49.3`；
predictor-last+encoder-last后为`83.3/83.3`、`80.0/86.7`、`77.3/78.7`、`66.0/55.3`。
layout用predictor-first+encoder-last可达`78.7/70.7`，说明最佳更新层依shift变化。

PushT horizon200（Table 2）：global straight GD/CEM `84/74→85.3/81.3`，额外`.03s`；spatial
`91.3/89.3→92/93.3`，额外`.01-.02s`；DINO-WM `68/86.7→70/90`。PushObj中seen T
约`50→88`，unseen square约`20→51`；固定16k数据、四shape预训练后unseen adapt51.9，单shape
预训练45.8，说明在线适应仍受预训练覆盖限制。LoRA和layer/buffer消融没有统一最优配置。

**优点、局限与 TDWM 关系。** 它使用真实执行outcome做闭环自监督，视觉和动力学shift都测，
不需reward/expert。它不是RL，而是online system identification；采取坏动作后才获得信号，未解
安全探索；episode reset、target坐标可能随encoder更新漂移、simulator shift有限，缺失特征无法
凭一步更新恢复。“执行后校准再replan”已不是空白。TDWM若用RL，应把reward/TD error/
uncertainty用于选样或约束更新，并与纯prediction AdaJEPA比较，而不是复述plan-execute-update。

#### Hi-LeWM（`F`）

**准确题名、作者与发表状态。** Niccolò Caselli、Francesco Massafra、Samuele Punzo、Salvatore
Lo Sardo、Ippokratis Pantelidis、Sathya Kamesh Bhethanabhotla，*Mind the Gap: Promises and
Pitfalls of Hierarchical Planning in LeWorldModel*；[arXiv:2607.12547](https://arxiv.org/abs/2607.12547)。
PDF 明示被 **WM@Booth 2026 Workshop on World Models** 接收，因此旧条目“未核到接收”错误；
准确状态是 workshop paper，不是主会。

**故事、模块与训练。** 冻结pretrained LeWM encoder、action encoder和low-level predictor。
从trajectory按temporal gap取`(z_t,action chunk,z_{t+k})`，macro-action Transformer将变长primitive
chunk压成默认32-D `ell`，high-level AdaLN predictor学`(z_t,ell)->z_{t+k}`的latent MSE。高层
CEM搜连续macro-actions、递归预测subgoals，第一个subgoal再交给冻结low-level LeWM CEM；高层
没有SIGReg，梯度也不进入低层（Secs. 2-3）。

**核心失败诊断。** PushT offset25/50/75，flat LeWM为`94±2/52.7±5/18±2`，naive hierarchy
`89.3/38.7/15.3`。offset50若给真实future latent作oracle subgoal可达`73.3±4.2`，说明低层能执行
好subgoal。Table 1：teacher stage1/stage2 forecast error `.081/.104`，open-loop true macro
`.081/.216`，CEM-selected macro `.347/.011`。CEM找到terminal看似极准、intermediate严重错误的
OOD macro-action，是典型model exploitation，而不只是representation collapse。

**修正与证据。** empirical macro bank用训练action chunks的codes作anchor，只搜
`ell=ell_bank+lambda_res epsilon`，并保留zero-res candidate。offset50 empirical online48.7、
staged64，相对flat +11.3；offset75 online32.7，相对flat +14.7，staged22；offset25无可靠收益。
representation ablation offset50/75：continuous-random32 `46/20`，fixed32 `42/20`，fixed8
`42/20`，VQ16 `38/20`，VQ128 `44/34`。高层/低层CEM预算可到`1200x60/1200x30`，不是轻量
免费提升；Cube结果也较混合。

**优点、局限与 TDWM 关系。** 论文可贵之处是报告负结果，用oracle定位subgoal generation，
把support mismatch说清楚。局限是两任务、三seed、best sweep、macro bank依赖demo support，
staged无反馈，高层MSE不等于control feasibility。它提示RL长期信息最有价值的角色可能是学习
低层controller的subgoal occupancy/support；若TDWM用successor representation约束LeWM
hierarchy，Hi-LeWM是必需对照。

#### Temporal-Distance-JEPA（`F`）

**准确题名、作者与发表状态。** Jiaxin Bai、Jiaxuan Xiong，*Temporal-Distance JEPA: Plan-Aware
Representation Learning for Latent World Model Predictive Control*；
[arXiv:2607.25337](https://arxiv.org/abs/2607.25337)（本次核读v2），当前为预印本。

**故事。** LeWM训练next-latent prediction，CEM却需按goal progress排名imagined futures；latent
L2只是偶然几何。reward-free demonstration有先后顺序，可挖“沿观察路径从i到j用了j-i步”的
directed temporal cost。作者区分两种部署：TwoRoom/Reacher这类topology task直接用temporal
cost；PushT/Cube接触任务只用temporal loss塑造checkpoint，planning仍用L2（Secs. 1、3.1）。

**cost head和完整目标。** MRN定义
`d_psi(z_i,z_j)=||phi_sym(z_i)-phi_sym(z_j)||+max_k ReLU(phi_asym,k(z_i)-phi_asym,k(z_j))`，
兼有非负对称配置项与方向残差（Eq. 5）。同轨迹`i<j`用Smooth-L1回归`j-i`；跨trajectory置换
goal用margin hinge。作者明确说observed path length只是shortest path上界，cross-trajectory会有
false negative。保留LeWM一步loss和SIGReg，再从真context递归rollout expert actions H=5，
`L_roll=H^{-1}sum_h||z_hat_{t+h}-sg(z_{t+h})||^2`。总目标四项权重默认
`lambda_roll=.5,lambda_td=1,lambda_sig=.09`。ViT-tiny约15M、latent192、history3；MRN hidden512、
sym/asym各128；AdamW `5e-5`、10epochs（Sec. 3，App. C）。

**推理。** CEM/iCEM rollout H=5后最小化`d_psi`；Reacher把terminal与trajectory-mean以`.3`
混合。PushT/Cube主协议改用同一checkpoint的L2。因此这些 manipulation增益不能归给deployed
temporal head。默认300 candidates，30 refinements（Cube10），50 episodes，goal offset25；主表
用10 independent plan seeds。

**锁定结果和消融。** Table 2 LeWM/RC-aux/TD-JEPA：TwoRoom
`97.4±1.3/98.6±1.0/100±0`，Reacher `96.0±1.9/96.8±1.4/97.0±2.4`，PushT
`83.6±3.2/81.4±1.9/86.0±4.2`，Cube `68.0±2.8/81.6±2.8/82.2±2.9`。PushT三训练seed的
full在blend/d_psi/L2/iCEM为`82.7/69.3/85.3/65.3`；symmetric temporal head
`56.7/55.3/57.3/43.3`，无cross hinge`72/68.7/77.3/49.3`，无rollout
`62/56.7/60/44`（Table 4）。三项均有贡献。

同checkpoint十seed的cost matrix（Table 11）：directed cost在四任务
`100/97/69/77`，L2 `99/95/86/82.2`，blend`.1` `100/95.6/83.4/81.2`。PushT上temporal head
与gap的Spearman `.914±.018`，L2 `.785±.016`，但L2 control更好；失败集中在contact后（Tables
6-7）。Reacher若terminal-only，TD-JEPA/LeWM `74/86`；progressive aggregate才`96/94`
（Table 10），planner aggregation同样影响结论。

**优点、局限与 TDWM 关系。** 优点是训练cost与plan cost对齐、有方向性、对LeWM/RC-aux用
locked protocol，并诚实报告contact任务反例。局限是step count受demo效率/support限制，跨轨迹
假负例，任务依赖地选择`d_psi`或L2，并非统一cost。它已覆盖reward-free temporal mining；
TDWM必须超出`j-i`，用offline RL/policy occupancy学可跨轨迹stitch的长期算子，并解释如何保留
contact geometry，否则与此工作高度重叠。

#### Temporally Centered SIGReg / TC-LeWM（`F`）

**准确题名、作者与发表状态。** Chang Liu、Fei Suo、Yanzhou Jin、Yusuke Iwasawa、Yutaka
Matsuo、Yaonan Zhu，*Temporally Centered SIGReg Improves Multi-Task LeWorldModel Learning:
From Analysis to Method*；[arXiv:2607.26924](https://arxiv.org/abs/2607.26924)（本次核读 v2）。
截至调研日未核到 proceedings 或正式接收记录，准确状态是 arXiv 预印本。

**故事。** 单任务 LeWM 用 full-marginal SIGReg 很稳，但多任务 latent 天然是 mixture：task、场景
和进度会形成慢变 cluster center，动作动态是较快的 cluster 内变化。强迫整个 mixture 变成一个
`N(0,I)`，可能通过压近 centers 来降低 Epps-Pulley loss；总体方差没有坍塌，task/state 却会
alias，使下游 policy 面对相似 latent 要输出不同动作。作者因此不删除 SIGReg，而是只高斯化
时间窗去均值后的 fast residual（Sec. 3-4）。

**模型、数据流和目标。** 两相机 view 经共享、从零训练的 ViT-S/16（12层、width384、6 heads），
linear/projector 后每个 view 得 512-D CLS；6层 block-causal predictor 用 AdaLN-zero 注入 frame
gap 内的 7-D action stack。每个 view 独立计算
`zbar_t=|W_t|^{-1}sum_{s in W_t}z_s`、`r_t=z_t-zbar_t`。Raw LeWM 的目标是
`L_pred+lambda SIGReg(Z)`，TC 只改成 `L_pred+lambda SIGReg(R)`；网络、J=1024 projections 和
gradient path 相同，encoder/predictor均联合更新（Eqs. 7-15，App. A.1）。若窗口内输出全常数，
`R=0`仍被EP统计量惩罚；随机noise虽可能过SIGReg，却难被dynamics预测。这个组合防窗口内
坍塌，但并不显式保证不同窗口的 centers 分开。

**分析、推理协议和具体结果。** balanced homoscedastic Gaussian-mixture 分析用 K=2/5/10/20、
每个K 500组随机centers、`rho=std(mu_k)/sigma`从0扫到4；raw EP的径向梯度多数收缩centers，
residual EP对rho近乎不变（Fig. 2）。真实 LIBERO-Long episode-cluster 的rho为 Raw `.74`、
TC `2.54`。这里没有 latent MPC：预训练后冻结encoder，由79.3M、12层flow-matching DiT BC
policy读取两相机共34 tokens，不输入proprio，预测8-step action chunk并open-loop执行8步；
checkpoint用validation inverse-dynamics metric选择。

LIBERO Spatial/Object/Goal/Long 的10-task共享训练（3 training seeds，每seed 3 eval seeds、每任务
50 rollouts）中，Raw为`61.6±4.3/46.4±12.8/74.8±6.2/29.9±8.3`，平均53.2；TC为
`68.6±1.8/87.8±3.2/86.0±1.1/51.8±9.2`，平均73.6（Table 1）。Long从单任务到10任务：
Raw `40.0→29.9`，TC `49.6→51.8`（Table 2）。40-task统一模型Raw平均44.4、TC73.5，TC四suite
为`72.3/83.6/84.9/53.3`（Table 3）。window消融W=4/8/32/整episode为
`44.7/51.8/48.2/53.4`，均优于Raw29.9；W=1 residual恒零并坍塌（Table 7）。

**优点、局限与 TDWM 关系。** 优点是发现了 anti-collapse objective 自身的多任务结构冲突，
只换regularizer target且matched control、尺度和window消融完整。局限是mixture分析依赖
balanced/homoscedastic和“局部均值等于慢语义”的假设；下游是大BC policy而非LeWM CEM，
未证明future rollout或planner同步改善，iDM选checkpoint也偏向action-readable latent。对TDWM，
RL goal/successor很可能产生多模态低频centers；可让TC-SIGReg保护局部动态residual，再由RL塑造
task/progress center，但必须正交消融full-SIGReg、TC-SIGReg、RL及其交互。

#### PhyLatent（`F`）

**准确题名、作者与发表状态警报。** Xi Zeng、Haojie Ren、Ziying Song，*PhyLatent: Learning
Dynamics-Relevant Representations for JEPA World Models*；
[arXiv:2608.05720](https://arxiv.org/abs/2608.05720)，首次提交于2026-08-06。PDF每页印有
“Published as a conference paper at ICLR 2025”，但论文引用多篇2026工作，且未核到对应官方
ICLR/OpenReview记录；这个时间关系不可能成立。本文只能列为 **arXiv预印本**，并明确把该页眉
视为疑似模板或元数据错误，不能据此声称已在ICLR 2025发表。

**故事和三类失效。** 作者把“非坍塌”细分为三个局部控制关系：同一物理状态换颜色/亮度后
latent变化大于真实状态变化（physical-invariance failure）；物理远状态折进latent近邻
（identifiability failure）；不同action的真实future分离，而model预测的branches却压在一起
（counterfactual-dynamics failure）。全局SIGReg和逐transition MSE都不直接排除这些情况
（Sec. 3，Table 1）。

**模型、数据流、五项损失和梯度。** backbone是LeWM式从零ViT-Tiny、latent192、history3和
6层predictor；baseline为stop-gradient future target的`L_pred`加权重`.09`的SIGReg。部署图保持
encoder/predictor/CEM不变，约19.6M训练图另加五条路径（Sec. 4，Fig. 3）：SVIC仅增强分支追随
原latent，约束brightness/color invariance；PSG让共享state head从encoded和predicted latent回归
z-score simulator state；FRA以projection和action-query cross-attention对齐predicted/encoded
future，targets stop-gradient；CASC置换batch actions并加noise，用action-distance控制的hinge
拉开counterfactual branches；LD给stop-gradient future latent加Gaussian noise，denoiser结合
predicted future/action/noise level预测噪声，梯度仍经predicted future回到world model
（Eqs. 11-27）。五个loss weight按任务单独调。物理targets也是任务定制：Cube 28维含joint、
velocity、EE/cube pose/contact，TwoRoom仅x/y等（App. Table 7）。

**数据、推理和结果。** 使用stable-worldmodel数据：Cube 2.01M transitions/10k episodes、
TwoRoom .921M/10k、Reacher 2.01M/10k、PushT 2.337M/18,685；90/10 split，batch32、10epochs，
单RTX5080。CEM horizon5/receding5、100 candidates/top10；goal为同trajectory后25步，3 seeds、
每seed100 episodes（Sec. 5.1-5.2）。Cube三种failure从`15.60/6.71/8.41`降至
`7.53/.95/4.62`，success `70.0±4.0→78.1±2.8`（Table 10）。TwoRoom
`81.0±6.24→98.0±1.0`，Reacher `78.33±2.08→79.33±3.51`，但PushT
`77.67±.58→75.33±2.08`反而下降（Table 4）。PushT三项诊断仍从`1.33/1.15/4.25`改善到
`.20/.52/2.38`（Table 12），所以作者定义的物理failure metrics不是planning成功的充分条件。
Cube去PSG+FRA/去SVIC/去CASC+LD的success为`74/77/73.33`，full为78.1（Table 5）。

**优点、局限与 TDWM 关系。** 优点是针对appearance、global folding和action consequence，
而非只报rank；同一inference图和planner，也诚实报告PushT反例。局限是使用privileged simulator
state、人工augmentation、任务专用变量和五组权重，不是self-supervised/reward-free物理发现；
counterfactual actions可能离data support，threshold也是作者自定义。它几乎直接覆盖“SIGReg后仍
物理坍塌”的故事。TDWM应把PhyLatent视为privileged upper bound，检验RL successor/value能否在
不看真state时减少同类failure，并用PushT验证诊断是否真的转化为planning，而不能只复用其术语。

#### PSG-JEPA（`F`）

**准确题名、作者与发表状态。** Haodong Yan、Jiaguan Zhu、Mingyuan Jia、Ruiqing Yin、Junjie
He、Zhide Zhong、Junfeng Li、Jinxuan Lu、Hengtao Li、Tianran Zhang、Jiayi Chen、Wenxuan
Song、Wen Chen、Yuxiang Gao、Haoang Li，*Is Forward Prediction Enough? Physical State
Grounding for JEPA World Models*；[arXiv:2608.06799](https://arxiv.org/abs/2608.06799)。截至调研日
为arXiv预印本；“PSG-JEPA”是简称，不能代替准确题名。

**故事和模型。** forward JEPA只要求`(z_t,a_t)`可预测next latent，不保证单个latent能读出机器人
状态，也不保证一对endpoint latent能读出实际状态变化。作者利用机器人轨迹通常自带的
proprioception作训练期privileged grounding，部署时删除heads。backbone沿用LeWM encoder和
causal action predictor，C=3 context、T=4 frames；teacher-forced one-step `L_fwd`加原SIGReg
（`lambda_reg=.09`）。static head从每个`z_t`回归joint angles、gripper、EE pose；dynamic head对
所有k=1,2,3及valid `(z_t,z_{t+k})`回归固定维`Delta q=q_{t+k}-q_t`，各horizon等权。总目标是
`L_JEPA+.1(L_static+L_dynamic)`，两个head的梯度在训练时更新encoder，推理时均移除
（Eqs. 1-7）。作者不用inverse action sequence，是因为长horizon动作多解且维度增长；但net joint
change也不覆盖物体/contact全部状态。

**probe、planner和rollout证据。** Cube单latent linear/MLP Pearson r中，LeWM→PSG的JointPos
`.71/.69→.83/.81`，EE-yaw `.08/.08→.94/.98`；endpoint pair的JointVel为
`.68/.66→.75/.75`，Action为`.74/.76→.80/.86`（Tables 1-2）。planner协议不是原LeWM CEM：
冻结encoder后统一训练3层MLP GC-IDM。Cube full-data、5/10/25/100epochs的LeWM为
`80.7/83.3/84.2/89.7`，PSG为`95.0/92.7/94.5/98.7`；PSG 10epoch低于5epoch，不能说单调
（Table 3）。原predictor在logged actions上递归30 model steps，Cube MSE
`.1488→.0485`、Scene `.1608→.0982`；5步为`.0093→.0046`、`.0269→.0208`，说明收益不只来自
下游GC-IDM。

**policy、真机与消融。** LIBERO-Goal中action head与encoder共同fine-tune，LeWM
`77.7±.5`、ActionIDM `82.6±2.2`、DINO `80.1±5.3`、PSG `85.3±3.9`（Table 4），因此这里
混合了预训练表示与下游joint fine-tuning。三项双臂真机任务各50 trials：Bread `84 vs62`、
Plate `74 vs58`、Pour `80 vs60`，均值79.3对60.0，但未给多训练seed置信区间（Fig. 4）。Cube
5-epoch planner/LIBERO消融：LeWM `80.7/77.7`；无transition `81.3/80.3`；无state
`93.3/80.0`；adjacent-only `93.5/81.5`；first-last `93.7/81.2`；full `95.0/85.3`（Table 5）。

**优点、局限与 TDWM 关系。** probe、world-model rollout、policy和真机构成较完整证据链，
有ActionIDM对照且部署无head成本。局限是proprio/joint labels属于privileged监督，不是RL或
self-supervision；GC-IDM改变planner，policy又joint fine-tune，动态target只ground机器人关节，
真机规模较小。对TDWM，它是physical-grounding监督上限：RL successor/occupancy若要声称提供
同类物理可读性，需同时与PSG、ActionIDM比较，并分别在原CEM和统一下游policy协议验证。

#### Metric Non-Collapse（`M`）

**准确题名、作者与发表状态。** Alain Bensoussan、Minh-Nhat Phung、Minh-Binh Tran，*Metric
Non-Collapse in Learned World Models for Control: Approximation Theory, Finite-Sample Geometric
Guarantees, and Deterministic Planning Transfer*；
[arXiv:2608.07265](https://arxiv.org/abs/2608.07265)，当前为math.OC预印本。本文核读定义、主定理链
和全部数值实验，但没有逐行审计70余页中每个spline/covering proof，故保持`M`而不把证明细节
写成完全复核。

**故事、设定与反例。** 对deterministic系统`s'=G(s,a), x=H(s)`，仅要求
`Phi(H(G(s,a)))≈F(Phi(H(s)),a)`不能排常数`Phi=F=z0`的零loss（Prop. 5.1）。full-rank
covariance也不够：一维`psi(s)=Cs^2`把`s,-s`折叠仍有正方差；二维`(Cs_1^2,Ds_2)` covariance
正定仍不injective（Prop. 5.3）。控制需要点态co-Lipschitz下界，而不是只匹配边缘分布。论文先
假设存在有限维紧凸Euclidean observable factor，observation在该factor上bi-Lipschitz，训练分布有
lower-Ahlfors coverage，模型类有统一`C^{1,1}`/Lipschitz budget；这些A1-A4是强前提，不是算法
从pixels自行发现的事实（Secs. 3-4）。

**损失和梯度。** prediction residual外，local hinge对state和单位tangent `(s,v)`惩罚
`[kappa-||D(Phi∘H)(s)v||]_+^2`，阻止任何切向方向的微分坍塌；global hinge对物理距离至少rho的
pair惩罚`[alpha-||psi(s)-psi(s')||]_+^2`，阻止远状态folding（Sec. 6）。它直接更新encoder，
无需decoder/inverse head；但训练必须拿到真实state metric和tangent direction，绝非pixels-only
self-supervision，规划时仍使用learned dynamics/cost和deterministic optimizer。

**理论真正给了什么。** 在上述coverage、smoothness、metric sampling以及norm-constrained
tensor-product B-spline可逼近精确noncollapsed realization等条件下，Theorem 7.15要求
`epsilon_stat<P*`且lambda超过由approximation/statistical/optimization误差决定的阈值；随后每个
近似empirical minimizer以高概率具有co-Lipschitz常数
`c_*=min(kappa/4,alpha/(2 diam S))`和统一semiconjugacy误差eta。若latent transition的Lipschitz
常数为`L_F`，rollout误差按`eta C(T,L_F)`传播，再加cost-head和planner误差得到finite-horizon
optimizer-transfer bound（Cor. 9.2-9.4）。`L_F<1`可有界，`=1`时terminal约O(eta T)、累计cost
约O(eta T^2)，`>1`可指数增长。这是充分条件链，不表示普通ViT训练自动满足阈值。

**实验和反证边界。** toy实验A的12 seeds中，ordinary init下pure prediction/covariance/
local-only/global-only/local+global发生非injective的次数为`10/1/1/0/0`；但故意folded init时plain
local+global只逃出3/12，加upper directional term才9/12，暴露one-sided hinge的optimization/
scale问题（Tables 7-9）。一维系统实验B用n=2048、lambda=.03可得dense local defect0、lower
chord`.68`、uniform residual`.02`，但实际`epsilon_stat/P*>10^5`，所有tested n都不满足定理
条件，作者明确没有数值验证certificate（Table 10）。非像素pendulum实验C中pure prediction虽
uniform residual最小`.01`，却collapsed且paired planning-cost gap约`1.41/1.54/1.55`；proposed
hinge为`.20/.18/.32`，并不优于所有其他regularizer。理论proxy与实测worst gap的排序相关为
`-.06`且极松（Tables 11、13）。

**优点、局限与 TDWM 关系。** 优点是严格证明variance/covariance不能排global folding，并把
approximation、统计、optimization、cost和planner误差分开；作者也诚实报告bound未满足、hinge
在benign benchmark并非赢家。局限是deterministic、instantaneously observable、强coverage/
smoothness/bounded class与privileged metric supervision，实验仅1D和smooth pendulum，不能直接
外推LeWM。TDWM不能再用effective rank或SIGReg声称“物理non-collapse”；若用RL替代真state
metric，应把policy occupancy/value解释为行为度量的近似lower bound，并实测local branch
separation、global folding、uniform rollout和planner transfer，同时明确coverage/identifiability缺口。

### 14.5 RL 梯度怎样帮助表示与无解码器世界模型

#### SPR（`F`）

**准确题名、作者与发表状态。** *Data-Efficient Reinforcement Learning with Self-Predictive
Representations*，Max Schwarzer、Ankesh Anand、Rishab Goel、R. Devon Hjelm、Aaron Courville、
Philip Bachman，ICLR 2021 正式论文；[原文](https://arxiv.org/abs/2007.05929)。

**故事与问题。** 它不是训练一个用于规划的 world model，而是问：Atari 只有 10 万次交互时，
能否让 Q-learning 的视觉编码器更快抓住“哪些画面变化可由动作预测”？Rainbow 的 RL loss
负责学任务价值，SPR 是共享 encoder 上的自监督辅助任务，负责保留短期可预测动态（Sec. 1-2，
PDF pp. 1-5）。

**模型、数据流与损失。** 在线编码器 `f_o` 编码增强后的当前帧；卷积 transition `h(z,a)` 按真实
动作递归预测未来 `K=5` 个 latent。另一条 EMA target encoder `f_m` 独立编码相应 future frames。
两端经 projector，在线端再经 predictor，用 Eq. (4) 的归一化 cosine loss逐步对齐；target 分支
stop-gradient。该损失与 Rainbow distributional Q loss 相加并共同更新在线 encoder。transition
训练后不参与选动作或规划，测试仍是普通 Q 网络（Sec. 2.2-2.3、Fig. 2，pp. 4-5）。

**实验与具体证据。** Atari100k 共 26 games；SPR 10 seeds。Table 1（p. 7）中 SPR mean/median
human-normalized score 为 `0.704/0.415`，DrQ 为 `0.357/0.268`，无 augmentation 的 SPR 为
`0.463/0.307`，并有 7 个游戏超过人类。Table 2（p. 8）中一步预测降到 `0.570/0.301`，
non-temporal 为 `0.507/0.271`，去 projector 为 `0.437/0.171`，未归一化 quadratic loss 几乎崩到
`0.047/0.040`。无 stop-gradient 时 median 为 `0.278`，完整方法为 `0.415`（Sec. 5、Appendix
Table 7，pp. 8、17）。P100 上约 4.6h，controlled Rainbow 约 2.1h（Appendix Table 8，p. 18）。

**优点、局限与本项目关系。** 多步、归一化、target 和 projector 的消融较完整。但“RL 防坍塌”
只是作者对 `tau=0` 仍可训练的解释：实验没有把 augmentation、RL gradient 与其他正则正交拆开。
它没有跨奖励迁移、规划质量、物理状态 probe 或 OOD dynamics。SPR 只支持一个窄假设：RL
gradient 可能把 LeWM latent 推向任务相关可控因素；它不证明 RL 单独足以防 collapse，也不证明
所得表示是通用物理 world model。TDWM 应至少比较 LeWM-only、显式 anti-collapse、LeWM+RL、
LeWM+RL+anti-collapse，并同时测 rank、长期预测、规划和换 reward 后的冻结迁移。

#### Self-Predictive Representations for RL（`F`）

**准确题名、作者与状态纠正。** 这里实际对应 *Bridging State and History Representations:
Understanding Self-Predictive RL*，Tianwei Ni、Benjamin Eysenbach、Erfan Seyedsalehi、Michel Ma、
Clement Gehring、Aditya Mahajan、Pierre-Luc Bacon，**ICLR 2024** 正式论文；
[原文](https://arxiv.org/abs/2401.08898)。旧版写成“NeurIPS 2024”以及把简称当准确题名均不正确。

**故事与理论对象。** 作者从“RL 到底需要怎样的表示”出发，把三类 abstraction 排成包含关系：
`Q*` irrelevance 只保最优控制信息；self-predictive/model-irrelevance 要预测 reward 与 next latent；
observation-predictive belief 还要复原观测历史。Theorem 1 给出 `observation-predictive =>
self-predictive => Q*-sufficient`，因此自预测位于“足以控制”与“不重建所有像素”之间（Sec. 3，
pp. 3-5）。Theorem 2 又说明，端到端 `Q*` 学习加精确 latent prediction 时 reward prediction 可被导出，
所以实践算法不另设 reward model。

**模型、梯度与损失。** 实践是 DDPG/TD3 风格 actor-critic 加共享 encoder 和一步 transition：
`g(phi(h_t),a_t)` 回归 `stopgrad(phi(h_{t+1}))`。没有 decoder、projector、多步 rollout 或 planning。
严格 ZP 要匹配 latent next-state distribution，EZP 只匹配条件均值；实践的 deterministic L2 或
probabilistic f-divergence 是 ZP 上界，确定系统可精确，随机系统会有 double-sampling bias。
stop-gradient 的 stationary point 可满足 EZP，online 双端更新没有保证（Sec. 4，pp. 5-7）。

**理论保证的边界。** Theorem 3 只在线性 encoder/transition、连续时间梯度流、transition 每时刻
先优化到最优等强条件下证明 `phi^T phi` 守恒，从合适初始化推出不坍塌。它不是深度网络、
actor-critic 与 encoder 联合训练的全局收敛定理。

**实验。** MuJoCo 四任务、12 seeds、500k steps 中 ZP 通常优于 TD3，Humanoid 上三步 ALM
有时更强（Fig. 3，p. 8）。Fig. 4（p. 8）用 1.5M steps、latent 50、batch 512 展示 online target
的有效 rank 明显下降，而 detached/EMA 较稳。加入 `2^4` 到 `2^8` 维 Gaussian distractor 后，
ZP 比 observation prediction 更稳（Fig. 5，p. 9）。20 个稀疏 MiniGrid POMDP、9 seeds、4M
steps 中 OP 最强，ZP 优于 R2D2，EMA rank 最稳（Sec. 5.3、Appendix Fig. 14-15）。

**优点、局限与本项目关系。** 这是“RL value gradient 与 latent prediction 互补”最重要的概念
依据；同时论文明确没有 pixel 实验，线性理论也未覆盖实际联合优化。它不能证明 RL 单独防坍塌，
更不能证明 self-predictive latent 等于完整物理状态。对 LeWM 的关键检验应是：RL gradient 究竟
提高 control sufficiency，还是破坏 reward-agnostic dynamics；必须同时测 frozen transfer 与 joint
fine-tuning。

#### DreamerPro（`F`）

**发表状态与主来源。** Fei Deng、Ingook Jang、Sungjin Ahn，ICML 2022，PMLR 162 正式论文；
[PMLR 原文](https://proceedings.mlr.press/v162/deng22a.html)。

**故事与模型。** Dreamer 的 pixel decoder 会迫使 latent 记住背景。DreamerPro 保留 DreamerV2
RSSM、reward/discount model 与 latent imagination actor-critic，却用 SwAV 式 prototype prediction
替换图像重建。两个 temporally consistent random-shift views 分别进入在线和 EMA encoder；目标
embedding 经 Sinkhorn-Knopp 得到均衡 assignment。RSSM state 投影预测 target assignment，在线
image view 也做 cross-view assignment（Sec. 3、Fig. 1-2，PDF pp. 2-4）。

**损失与防坍塌来源。** Eq. (6) 的 `J_SwAV` 让同帧两视图落到相同 prototype，Sinkhorn 的等量
约束直接防止所有样本选同一 prototype；Eq. (7) 的 `J_Temp` 让由历史和动作得到的 RSSM state
预测当前 prototype。Eq. (8) 将二者与 reward prediction 和 RSSM KL 相加，actor/critic 仍按
Dreamer imagination 训练。默认 2500 prototypes、维度 32、temperature 0.1、Sinkhorn 3 次、
EMA update fraction 0.05（Appendix Table 3，p. 12）。

**控制推理。** prototype/projector 只在训练 world representation 时提供目标，部署动作并不在
prototype 空间做 MPC。与 Dreamer 一样，actor 直接消费 RSSM state，critic 估计 imagined return；
behavior-learning 阶段冻结 world model，策略梯度不会回写 encoder/RSSM。

**实验与数字。** 六个 DMC visual tasks、1M environment steps、3 seeds、每次评估 10 episodes。
标准背景 Table 1（p. 5）中 Walker Run 为 `784`，Dreamer `737`。自然视频背景 Table 2（p. 6）
中 Cartpole `671` 对 Dreamer `126`，Finger `826` 对 `10`，Walker `394` 对 `35`；但 Cheetah
`349` 低于 TPC-Batch300 的 `477`，Cup `493` 低于 Dreaming 的 `553`。Fig. 6/Appendix Fig. 7
显示去掉 `J_SwAV` 或 `J_Temp` 都下降。64×64 时 DreamerPro 44 FPS、Dreamer 48；256×256
时 9 对 3（Appendix Table 6，p. 16）。

**优点、局限与本项目关系。** 它是 decoder-free Dreamer 的直接先例，但 anti-collapse 的明确
来源是 balanced prototypes、EMA 与 augmentation，不是 RL 自发作用。nearest-neighbor 仍会按
背景聚类；Atari 附录中还需 SPR/inverse-dynamics 辅助。TDWM 若声称 RL 可取代 anti-collapse，
必须与 Sinkhorn prototype 等计算量对照，并测试换 reward/换背景，而不能只看当前奖励回报。

#### MuDreamer（`F`）

**发表状态纠正。** Maxime Burchi、Radu Timofte，2024 arXiv 预印本；ICLR 2024 OpenReview
submission 状态为 **withdrawn**，未核实到后续 archival venue。
[arXiv](https://arxiv.org/abs/2405.15083)、[withdrawn 条目](https://openreview.net/forum?id=9pe38WpsbX)。
因此不能写成“ICLR 2024 已发表”。

**故事与数据流。** MuDreamer 从 DreamerV3 RSSM 出发，用控制相关预测替代像素 reconstruction。
observation encoder 得 `x_t`，RSSM 生成 deterministic `h_t` 与 stochastic `z_t`，state `{h,z}`
接 reward、continue 和 lambda-return value heads。另一个 action head 从当前 image feature 和前一
prior state 预测上一动作，在 sparse reward 下仍给控制信号。decoder 只接 stop-gradient state 做
可视化，不参与训练（Sec. 3、Fig. 2，PDF pp. 4-6）。

**损失、梯度与推理。** Eq. (2) 包括 reward、continue、value、previous-action prediction；再加
Dreamer KL balance：dynamics KL 用 stop-gradient posterior 教 prior，权重 0.95；representation KL
反向教 posterior，权重 0.05，并有 1 nat free bits。value 用 `lambda=.95, gamma=.997` 和慢 EMA
critic；representation network 内 BatchNorm 是作者明确指出的 collapse 防线。actor 在 15-step
imagination 中训练，连续动作走 dynamics gradient，离散动作走 REINFORCE（Sec. 3.2-3.3）。

**实验与消融。** 20 个 DMC visual tasks、1M steps、3 seeds。Table 1（p. 7）mean/median 为
`784.7/849.6`，DreamerV3 `739.6/808.5`，DrQ-v2 mean `677.4`。自然视频背景 Table 2（p. 8）
mean `517.0`，DreamerPro `445.2`、TPC `372.8`、DreamerV3 `83.6`。Table 3（p. 9）去 value
head mean `628.8`，去 action head `648.0`，两者皆去 `505.4`，去 BN `689.5`。Atari100k、
5 seeds 中 human mean `126%` 对 DreamerV3 `112%`，但 median `43%` 低于 `49%`（Table 4，
p. 14）。训练约 14h、15.3M 参数，DreamerV3 15h、17.9M。

**优点、局限与本项目关系。** 它直接证明 value/action/reward heads 可以帮助 reconstruction-free
Dreamer，但不证明 RL 单独防坍塌，因为 BN 明确承担关键作用。它依赖当前 reward，value target
来自 replay 旧策略且无 off-policy correction，未证明换奖励复用。LeWM 可借鉴可插拔 task heads，
但必须拆开 reward/value/action gradient，并用冻结表示跨奖励规划验证是否仍是通用 world model。

#### R2-Dreamer（`F`）

**发表状态与来源。** Naoki Morihira、Amal Nahar、Kartik Bharadwaj、Yasuhiro Kato、Akinobu
Hayashi、Tatsuya Harada，**ICLR 2026 Poster** 正式论文；
[原文](https://arxiv.org/abs/2603.18202)。早期 “under review” 状态已经过时。

**故事与模型。** DreamerPro 依赖 random shift，而小的任务关键物体可能被 shift/crop 丢失。
R2-Dreamer 要在无 decoder、无 augmentation 下防 collapse。它基本不改 DreamerV3 RSSM：image
encoder 得 `e_t`，RSSM state `s_t=(h_t,z_t)`，线性 projector 把 state 映到 `k_t`，再在整批
`B×T` 样本上计算 `k` 与 stop-gradient `e` 的 cross-correlation（Sec. 3、Fig. 2，pp. 3-5）。

**损失和梯度。** Eq. (5) 是 Barlow Twins objective：对角 `(1-C_ii)^2` 对齐同维信息，非对角
`alpha*C_ij^2` 去冗余，迫使不同维承载不同因素。它与 reward、continue prediction 和 Dreamer KL
相加；actor-critic 沿用 DreamerV3。`e` 在 BT target 端 detach，但 encoder 仍通过 posterior/RSSM
路径收梯度。默认 batch 16、sequence 64、BT weight 0.05、`alpha=5e-4`、imagination horizon 15
（Appendix Table 2，p. 19）。

**控制推理。** Barlow projector 是训练期 auxiliary head；真正执行时 actor 读取 RSSM state，
无需 projector、decoder 或图像增强。actor/critic 在 imagined trajectories 中更新，而 world-model
parameters 在 behavior-learning update 中冻结，所以 return gradient 不会偷偷成为另一条 encoder
anti-collapse 路径。

**实验与数字。** 20 DMC、50 MetaWorld、5 个作者构造的 DMC-Subtle；5 seeds、10 eval episodes。
DMC-Subtle 把 ball/string 宽度缩到原来的 `1/12`、pole 缩到 `1/20`，R2 显著领先（Fig. 4-6，
pp. 6-8）。Fig. 7 显示 DreamerPro 去 augmentation 接近 no-decoder 失败线；Fig. 8 显示 augmentation
对标准任务帮助小、对 Subtle 反而有害。RTX3080Ti 上 1M steps：R2 `4.4h`，同实现 Dreamer
`7.0h`，DreamerPro `10.4h`，官方 JAX Dreamer `6.6h`（Table 1，p. 10）。

**优点、局限与本项目关系。** 它是 LeWM anti-collapse 的直接强 baseline，证明显式 redundancy
reduction 可以替代 decoder/augmentation；却不证明 actor/critic gradient 防坍塌。动态干扰背景未测，
DMC-Subtle 又是针对 augmentation 弱点构造，标准 DMC/MetaWorld 主要给曲线而无总分数表。
TDWM 若声称 RL 让 LeWM 不 collapse，至少要与同 encoder/batch 的 Barlow/VICReg 类正则比较。

#### RLDP（`F`）

**发表状态与来源。** Pranaya Jajoo、Harshit Sikchi、Siddhant Agarwal、Amy Zhang、Scott Niekum、
Martha White，**ICLR 2026 Poster** 正式论文；此前有 RLC 2025 RLBrew workshop 版本。
[OpenReview](https://openreview.net/forum?id=jdL6WB5jHZ)。只标 workshop 已经过时。

**故事与两阶段设定。** RLDP 反问 zero-shot BFM 是否真需要复杂 successor/contrastive representation
objective。阶段一只用 reward-free offline `(s,a,s')` 学 encoder 和 latent dynamics；阶段二冻结
encoder，再训练 successor-feature BFM 与参数化 policy。因此 RLDP representation 本身不是靠 RL
梯度学到，RL/Bellman learning 在冻结后的第二阶段（Sec. 3-4，PDF pp. 3-6）。

**模型与损失。** 从长度 `H=5` 的片段开始，`phi(s_0)` 经 action-conditioned latent transition
连续 rollout，回归 slow target encoder 的未来 features。作者发现 stop-gradient/target 仍让不同
state cosine similarity 缓慢升高，即 mild collapse；于是把 encoder 和 prediction 归一到半径
`sqrt(d)` 的球面，并加 Eq. (7) `L_r=E[phi(s)^T phi(s')]`，总损失 `L_d+lambda L_r`（Eq. 6-8，
pp. 5-6）。随后固定 `phi`，用 successor-measure Bellman/contrastive loss训练 `F`，policy 最大化
`Q=F^T z`。该 pairwise signed-dot regularizer 比完整 covariance identity 更弱。

**实验与数字。** ExORL RND data 的 Pointmass/Cheetah/Walker/Quadruped，各 4 tasks；latent 512，
representation 2M updates、policy 3M、4 seeds。Table 1（p. 7）显示并非全胜：Walker Flip RLDP
`492.94`，FB `977.08`；Quadruped Jump RLDP `733.32`，FB `567.27`；Pointmass Top Right RLDP
`795.47`，FB `550.84`、PSM `666.00`。又测 358-d observation/69-d action 的 SMPL Humanoid、
45 tasks，以及低覆盖 D4RL 六任务、10 seeds；低覆盖下 RLDP 在 5/6 上超过对照。`lambda=.01`
显著好于 0，固定 H=5 时 `lambda=1` 最好；球面归一化也关键（Fig. 3、Table 2、Appendix A.4）。

**优点、局限与本项目关系。** RLDP 是一个重要反例：无需把 RL gradient 灌进 encoder，简单
reward-free dynamics 加显式 anti-collapse，再在 frozen representation 上学 policy 已经很强。实验
主要是 state 而非 pixel，H=5 仍局部，zero-shot 只覆盖 feature span 内奖励。它应是 TDWM 的
“LeWM frozen + RL head”首要 baseline；新方法需证明 joint RL gradient 的收益超过冻结方案且不
缩窄 reward span。

### 14.6 Stable World Model 中的决策基线

#### TD-MPC2（`F`）

**发表状态与来源。** Nicklas Hansen、Hao Su、Xiaolong Wang，ICLR 2024 正式论文；
[原文](https://arxiv.org/abs/2310.16828)。

**故事、模型与数据流。** TD-MPC2 把局部 latent world model、TD value 与 MPC 做成可扩展统一
系统。encoder `h` 得 latent，dynamics `d(z,a)` 预测 next latent，reward head 给即时 reward，五个
Q heads 给 rollout terminal value，stochastic policy prior 提议动作；multi-task 版另加 task embedding。
在线执行用 MPPI：采样 action sequences，latent model 累计 reward、末端接 Q，并混入 policy
trajectories 和上一时刻 warm start（Sec. 3、Fig. 2，PDF pp. 3-5）。

**损失与真实梯度路径。** Eq. (3) 对 replay H-step 片段逐步求和：latent consistency 回归
stop-gradient next-observation latent；reward 用 two-hot discrete cross-entropy；value 用 TD target
`r+gamma Q_bar(z',pi(z'))` 的离散回归，远期以 `lambda^t` 降权。Eq. (4) 的 max-entropy actor
**只更新 policy**，不把 actor loss 传进 encoder/dynamics。因此 world representation 收到的是
consistency、reward 与 Q supervision，而不是笼统的“actor gradient”。SimNorm 将 latent 分组
softmax，主要是稳定性归纳偏置，不是物理可辨识保证（Sec. 3.2-3.3、Appendix A，p. 16）。

**实验与数字。** 104 tasks：39 DMC、50 MetaWorld、5 ManiSkill2、10 MyoSuite；3 seeds、共享
超参。10 个 pixel DMC 与 DrQ-v2/DreamerV3 相当而非全面领先。80-task scaling 用 545M transitions：
1M/5M/19M/48M/317M 参数的 normalized score 为 `16.0/49.5/57.1/68.0/70.6`，317M 模型约
33 RTX3090 GPU-days（Fig. 7、Table 1，p. 7）。Fig. 9（p. 9）中 multi-task policy-only/planning/
combined 为 `42.2/53.7/54.2`；no norm/SimNorm/LN+SimNorm 为 `46.8/51.0/54.2`；continuous/
discrete regression 为 `49.6/54.2`；5 Q/10 Q 为 `54.2/57.0`。

**优点、局限与本项目关系。** 它有强闭环控制证据并用统一 MPC 真正检验 latent model；但它是
online、reward-dependent、主要连续控制与局部 horizon，不是 zero-shot 换任意 reward 的模型。
对 LeWM 最有用的是可审计地拆分 consistency、reward、Q 三路梯度与统一 planner。公平比较必须
单独切断这些梯度，并报告 online interaction、reward labels 和 MPPI compute，不能把 TD-MPC2
当 reward-free LeWM 的同条件替代。

#### GCBC / GCSL（`F`）

**发表状态纠正与来源。** 对应论文是 *Learning to Reach Goals via Iterated Supervised Learning*，
Dibya Ghosh、Abhishek Gupta、Ashwin Reddy、Justin Fu、Coline Manon Devin、Benjamin Eysenbach、
Sergey Levine，**ICLR 2021 Oral**；[OpenReview](https://openreview.net/forum?id=rALA0Xo6yNJ)、
[arXiv](https://arxiv.org/abs/1912.06088)。旧版写“ICLR 2020”不正确：2019 是预印本，正式年份
为 2021。BC 即 Behavioral Cloning；GCBC 是一般的 goal-conditioned BC，GCSL 是反复采集并
hindsight relabel 的具体算法，两者不能完全等同。

**故事、输入输出与训练。** 一条 trajectory 从 `s_t` 到了未来 `s_{t+h}`，其中 `a_t` 就可视为
“朝这个 future goal 的示范”。GCSL 将所有 `(s_t,a_t,g=s_{t+h},h)` 变成 supervised tuples，
最大化 `log pi(a_t|s_t,g,h)`；当前 policy 不断交互，新轨迹 relabel 后回 replay 再做 MLE。
没有 Q、reward model、dynamics model 或 planning（Sec. 3、Algorithm 1，PDF pp. 3-5）。

**理论与实验。** Theorem 3.1 给 on-policy GCSL objective 到真实 final-goal success 的下界，误差
含 `4T(T-1)alpha^2`；Theorem 3.2 在 deterministic dynamics、behavior full support 和小拟合误差下
给 `epsilon*T` 级 gap（Sec. 3.2，pp. 5-6），不等于 stochastic/offline 数据上的最优保证。实验含
FourRooms、Sawyer pushing、LunarLander、door、9-DoF claw，5 seeds；GCSL 多数最好或并列，
door 上 PPO 略强，原文主要给 curves 而无完整数表（Fig. 3-4，pp. 7-9）。固定初始数据、只保
on-policy 数据、仅 relabel 3 步 goal 都更差；显式 horizon 对 LunarLander 有益、对 Sawyer 探索
可能有害。

**优点、局限与本项目关系。** GCBC/GCSL 是 policy baseline，不是 world model。它证明不经 RL
value learning，hindsight supervision 已能注入 reachability，因此 LeWM+RL 的收益必须排除只是
更好的 relabel/BC。Stable World Model 的 fixed offline GCBC 也不等同迭代 GCSL。它可作为
“冻结 LeWM latent 后做 GCBC”的低风险基线，但不能回答反事实 actions 或物理 future fidelity。

#### GCIQL / IQL（`F`）

**发表状态与来源。** *Offline Reinforcement Learning with Implicit Q-Learning*，Ilya Kostrikov、
Ashvin Nair、Sergey Levine，ICLR 2022 正式论文；[原文](https://arxiv.org/abs/2110.06169)。GCIQL
通常是把 IQL 的 V/Q/policy 条件化到 goal，并不是另一篇 world-model 论文。

**故事与方法。** 离线 RL 的 max-Q 容易挑中数据外动作。IQL 整个 critic 训练只评估 dataset actions，
却用 upper expectile 让 `V(s)` 接近 behavior support 内较好动作的 Q，再把高优势数据动作加权
模仿出来（Sec. 3-4，PDF pp. 3-6）。Eq. (5) 的非对称平方 loss
`L_tau^2(Q_bar(s,a)-V(s))` 在 `tau>0.5` 时把 V 拉向 Q 分布上端，但不生成新动作；Eq. (6) 用
`r+gamma V(s')` 回归 clipped double Q；Eq. (7) 用 `exp(beta*(Q-V))` 做 advantage-weighted BC。
goal-conditioned 版本再用 future/random goals 构造 reaching rewards。

**理论、实验与数字。** 在有限、精确、bounded 且 `tau->1` 时，expectile 可恢复 behavior support
内 max，不是连续深网的全局保证。D4RL Table 1（p. 8）：IQL locomotion total `692.4`，CQL
`698.5`；AntMaze total `378.0`，CQL `303.6`；全表 `1070.4`，CQL `1002.1`。GTX1080 上
1M updates 少于 20 分钟，CQL 约 80 分钟。online fine-tuning Table 2（p. 9）IQL `408.2→597.7`，
CQL `182.8→241.0`。

**优点、局限与本项目关系。** 它稳定且避免训练时主动查询 OOD action，但最优性受数据 support
限制，expectile/beta 敏感，也没有 dynamics predictor。接 LeWM 时要区分 frozen probe 与让 IQL
gradient 进入 encoder：后者可能形成 reachability geometry，也可能抹掉当前 goal 无关的物理因素。
GCIQL 是检验 TD stitching 的 decision baseline，不能凭 success 推出 world-model fidelity 更高。

#### GCIVL / HIQL（`F`）

**发表状态与来源。** *HIQL: Offline Goal-Conditioned RL with Latent States as Actions*，Seohong
Park、Dibya Ghosh、Benjamin Eysenbach、Sergey Levine，NeurIPS 2023 正式论文；
[原文](https://arxiv.org/abs/2307.11949)。GCIVL 是 action-free goal-conditioned value learning
组件，HIQL 还包括 high/low-level policies，不能混称。

**故事、数据流与损失。** HIQL 用无需 action label 的 `(s,s',g)` 学 `V(s,g)`，再让 high-level
每 k 步输出 latent subgoal，low-level 输出 action。Eq. (4) 是 goal-conditioned expectile TD，不学 Q；
Eq. (6) 的 high-level AWR 用 `V(s_{t+k},g)-V(s_t,g)` 加权未来 subgoal，并输出中间 value layer 的
`phi(s_{t+k})`；Eq. (7) 的 low-level AWR 用 `V(s_{t+1},s_{t+k})-V(s_t,s_{t+k})` 加权真实 action。
Proposition 5.1 只在 deterministic dynamics、`V*` 精确可表示时说明 value representation 足以
low-level control；随机环境中的 action-free backup 有 optimistic bias（Sec. 4-5，pp. 4-8）。

**实验与数字。** 8 seeds。Table 1（p. 9）：AntMaze large-diverse HIQL `88.2`，GC-IQL `50.7`，
HGCBC `63.9`；ultra-diverse `52.9` 对 GC-IQL `21.6`；Kitchen partial `65.0` 对 `39.2`；CALVIN
HIQL `43.8`，GC-IQL `7.8`，该行 strongest baseline GCBC 为 `17.3`。Pixel Table 2：Procgen500
test `64.5` 对 GC-IQL `49.5`，VisualAnt diverse `80.5` 对 `72.6`，Roboverse `61.5` 对 `31.2`。
只给 25% trajectories action labels 时 large-diverse `88.9`，ultra-diverse 从 `52.9` 降到 `38.2`
（Table 3，p. 10）。distant-goal test policy accuracy 为 `59.9`，GC-IQL `47.5`。

**优点、局限与本项目关系。** HIQL 强在长 horizon 和少 action labels，但它是 value geometry 加
hierarchical policy，不是任意 action-sequence world model。它提醒 TDWM：只用 goal success 测
“物理意义”很可能只是在测 reachability。应把 GCIVL/HIQL 当 frozen-LeWM probe，并另外报告
one/multi-step counterfactual dynamics，才能证明 RL 不只是塑造价值表示。

#### HILP（`F`）

**发表状态与来源。** *Foundation Policies with Hilbert Representations*，Seohong Park、Tobias
Kreiman、Sergey Levine，ICML 2024，PMLR 235 正式论文；
[PMLR 原文](https://proceedings.mlr.press/v235/park24g.html)。

**故事与两阶段模型。** HILP 从 reward-free offline data 学“时间距离几何”。阶段一将
`V(s,g)=-||phi(s)-phi(g)||`，用到达 goal 前每步 -1 的 goal-conditioned IQL/expectile TD loss
训练 `phi`（Eq. 6）；阶段二冻结表示，把单位方向 `z` 当 skill，用 intrinsic reward
`<phi(s')-phi(s),z>` 做 offline RL，训练 `pi(a|s,z)`。新 reward 可按 Eq. (8) 线性回归；goal 用
Eq. (9) 的归一化方向；直线跨不过障碍时，Eq. (10-11) 从 dataset states 选 midpoint 递归 planning
（Sec. 3-4，PDF pp. 3-6）。

**理论边界。** Theorem 5.1 要求全局 embedding/directional policy error 满足
`4 epsilon_e+epsilon_d<1`，且假设 deterministic MDP。真实 temporal distance 常不对称，L2 却
对称；一般 MDP 未必存在低失真 Hilbert isometry，discounted value 也不等于无折扣最短路。

**实验与数字。** ExORL 四域×四任务×四数据集×4 seeds，共 256 组合，HILP IQM 最强，pixel
版本也优于 FB/FDM（Fig. 4-5，pp. 7-8）。Goal Table 1（p. 9，8 seeds）：AntMaze large-diverse
HILP `46.0`，HILP-plan `64.5`，GC-IQL `56.0`；ultra-diverse `21.2→59.2`，GC-IQL `40.8`；
Kitchen partial HILP `63.9`、planning `59.7`。Hierarchical Table 2 平均 `51.3`，OPAL `45.7`，
但若干 individual tasks 更差。

**优点、局限与本项目关系。** HILP 已经直接做“用 RL value geometry 学通用表示再 zero-shot”，
所以 TDWM 的 novelty 不能只是这句话。可区分之处是 LeWM 是否保留 primitive arbitrary-action
counterfactual model、视觉物理 factors 和统一 MPC。HILP 是检验“长期 operator 是否真的比普通
temporal metric 多信息”的必要 baseline。

### 14.7 时序差分、后继表示与长期世界模型

#### `gamma`-Models（`F`）

**准确题名、作者与发表状态。** *`gamma`-Models: Generative Temporal Difference Learning for
Infinite-Horizon Prediction*，Michael Janner、Igor Mordatch、Sergey Levine，NeurIPS 2020 正式
论文；[NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/12ffb0968f2f56e51a59a6beb37b2859-Abstract.html)。

**故事与模型对象。** 单步模型反复 rollout 会累计误差；`gamma`-model 不问“第 37 步是什么”，
而直接生成从 `(s,a)` 开始、之后跟 policy `pi`，在几何分布随机时刻会看到的状态。Eq. (1)
定义 normalized successor occupancy
`(1-gamma) sum_{k>=1} gamma^(k-1) p(s_{t+k}|s,a,pi)`。因此它是 policy-conditioned 的长期
occupancy model，不是给测试时任意 action sequence 做反事实预测的 primitive dynamics
（Sec. 3，PDF pp. 3-5）。

**训练、损失与价值推理。** Bellman distribution Eq. (2) 的 target 是混合分布：概率
`1-gamma` 取真实 next state，概率 `gamma` 从 target `gamma`-model 在 `(s',a'~pi)` 再采样。
作者分别用 GAN/f-divergence Eq. (3) 和 normalizing-flow likelihood regression Eq. (4)，并配
EMA target。Proposition 1 说明精确 fixed point 是 normalized successor distribution。Eq. (6)
把 state reward 的 Q 写成 `1/(1-gamma) E_{s+~mu} r(s+)`；`gamma`-MVE 再把这一步长期跳跃
与短模型/critic backup 结合。Theorem 1 可以通过样本重加权从训练 discount 估计更大的 target
discount，但不能消除训练时 TD bootstrap error 与测试时多次跳跃 compounding error 的权衡。

**实验与具体证据。** 预测实验只在 Acrobot、Pendulum 等低维系统，数据来自 SAC policy
mixture；高 `gamma` 时 flow 明显比 GAN 稳（Fig. 3-4，pp. 6-7）。控制实验为 Acrobot、
MountainCar、Pendulum、Reacher，5 seeds；模型 discount `.8`、控制 target `.99`，一次
`gamma`-model jump 对比 5-step MVE。`gamma`-MVE 收敛速度约为 SAC 的两倍，并接近 MBPO
的 sample efficiency，但原文主要给学习曲线，没有可逐项抄录的完整数值表（Fig. 5，p. 8）。

**优点、局限与本项目关系。** 它首次清楚建立 generative TD 的无限时域分布对象，是 TD-Flow、
UHM、Jumpy 的直接源头。Sec. 6（p. 9）也明确承认没有图像或复杂高维控制实验，且 GAN/
bootstrapping 会不稳；模型还绑定 policy，换 policy 需条件化或重训。对 LeWM 更可信的借鉴是
保留短期任意动作 dynamics，再增加 policy-conditioned occupancy head。TDWM 不能把“TD 长期
world model”本身当新意，也必须区分 occupancy prediction 与真正的 action-sequence simulator。

#### Temporal Difference Flows（`F`）

**准确题名、作者与发表状态。** *Temporal Difference Flows*，Jesse Farebrother、Matteo
Pirotta、Andrea Tirinzoni、Rémi Munos、Alessandro Lazaric、Ahmed Touati，ICML 2025，PMLR
267 正式论文；[PMLR 原文](https://proceedings.mlr.press/v267/farebrother25a.html)。

**故事与模型。** `gamma`-model 的 Bellman target 是分布混合，但 GAN 或普通 flow matching
持续 bootstrap 自己的 endpoint sample 时会累积偏差。TD-Flow 把 successor distribution 的递归
结构写进 conditional flow matching；TD2 又不只 bootstrap 终点，而让 target vector field 沿整条
probability path 提供监督，目标是同时降低长 horizon error 和 gradient variance（Sec. 3-4，
PDF pp. 4-8）。模型输入 `(s,a,policy/task condition,flow time)`，输出把 base noise 搬运到
policy-conditioned geometric-horizon future-state distribution 的 vector field。

**损失、梯度与理论边界。** 普通 TD-CFM 用真实 next state 或下一状态 target-flow sample
构造 mixture target；coupled 版本复用同一个 base noise，给 immediate 与 bootstrap path 建立
耦合。TD2-CFM Eq. (8) 直接匹配 bootstrap target 的 vector field，而不是把整条 path 压成一个
endpoint regression；diffusion 版 Eq. (13) 同理。Theorem 1 给理想 path operator 在
Wasserstein-1 下的 contraction，Corollary 给 exact optimization 下的收敛；Theorem 2-3 的低
方差结论依赖理想 target 与 covariance 条件，不是有限数据深网训练的普遍保证。

**实验与具体数字。** 使用 ExORL RND 的 10M transitions，Pointmass、Cheetah、Walker、
Quadruped 和固定 TD3 policies；3M updates、batch 1024、3 seeds，预测 horizon 5 到 100。
Fig. 3 显示 horizon 100 时 TD2 相对 naive model 的 MSE 可低近四个数量级。Table 1（p. 9，
horizon 100）的 value MSE：Pointmass 为 TD2-CFM `8.74`、TD-CFM `355.56`、GAN `1257.26`；
Walker `28.35/225.27/3690.65`；Cheetah `135.22/228.77`；Quadruped `141.77/525.06`。
随后把模型接入 FB policy family：GHM 训练 8M updates，GPI 用 256 policies 乘 128 model
samples，每任务 100 episodes；coupled/TD2 flow 相对基础 FB 平均提高约 30% 以上，而直接
FB-GPI 反而退化（Sec. 5，pp. 8-11）。

**优点、局限与本项目关系。** 这是长期生成式 occupancy model 的强直接前作，能处理高 horizon
和多模态分布，并可做 reward-free value estimation。代价是 flow sampling 需要 ODE 积分、训练量
大，实验主要是低维 state；对象仍是 policy occupancy，不是任意 primitive action rollout。TDWM
若加入 TD long-term head，必须比较 TD2-Flow，并明确优势来自视觉 latent、动作反事实、计算成本
还是 anti-collapse，而不能只说“TD 缓解长时误差”。

#### TD-JEPA（`F`）

**准确题名、作者与发表状态。** *TD-JEPA: Latent-predictive Representations for Zero-Shot
Reinforcement Learning*，Marco Bagatella、Matteo Pirotta、Ahmed Touati、Alessandro Lazaric、
Andrea Tirinzoni，**ICLR 2026 Oral** 正式论文；[OpenReview](https://openreview.net/forum?id=SzXDuBN8M1)。
所读 arXiv v1 日期是 2025-10-02，但继续把它写成“2025 arXiv 未发表”已不准确。

**故事与完整信息流。** 作者认为 one-step BYOL/latent prediction 只学习 behavior policy 的局部
动态，不足以 zero-shot 优化多种 reward，于是用 off-policy TD 把 predictor 训练成参数化 policy
的长期 successor features，再从 predictor 蒸馏相应 policy。数据是 reward-free offline
`(s,a,s')`；state encoder `phi:S->R^256` 服务控制输入，task encoder `psi:S->R^50` 定义线性
reward 空间；`T_phi(phi(s),a,z)` 输出 `psi` 空间的 successor feature，反向 predictor `T_psi`
输出 `phi` 空间；actor `pi(phi(s),z)` 选择最大化 `T_phi^T z` 的动作。测试时从少量 `(s,r)`
线性回归出 `z_r`，直接执行 `pi_zr`，并不做 MPC（Sec. 3、Algorithm 1，PDF pp. 3-5）。

**损失、梯度与防坍塌。** Monte Carlo Eq. (5) 原本要采 policy successor measure；TD 版
Eq. (7)/(9) 改为 `T(s,a,z) = target_encoder(s') + gamma target_T(s',a'~pi_z,z)`，因此可用
任意离线 one-step transition。两套交叉 encoder/predictor loss 让 state/task 表示互相监督，另配
EMA target。Algorithm 1 还显式加入样本间平方内积减自身范数的 orthonormal regularizer；actor
loss 为 `-T_phi(phi(s),a,z)^T z`。所以论文并没有证明“RL 自动防坍塌”，显式正交正则仍是核心
组成。

**理论应如何读。** Proposition 1 说明 MC predictor 的最优解是 successor-feature 条件均值；
Theorem 1/3 将理想 latent-predictive gradient 联系到 successor-measure factorization/TD loss；
Theorem 2 只在 tabular linear、连续时间、predictor 始终先优化到最优且适当初始化时给 encoder
covariance 守恒。Theorem 4 用 successor approximation error 上界单位范数 reward 的 policy-
evaluation error（Sec. 4，pp. 5-7）。正文还假设 uniform state、identity covariance、symmetric
policy kernel；Appendix C 放松后需要实践上难以 off-policy 采样的 action-conditioned backward
kernel（pp. 20-23），所以完整实际算法没有获得完全 relaxed 的理论保证。

**实验与数字。** 13 datasets、65 tasks：ExORL/DMC 四域和 OGBench 九域，同时测 state 与
64x64 RGB；DMC 5 seeds/20 eval episodes，OGBench 10 seeds/10 episodes。Table 1（p. 8）：
DMC-RGB 平均 `628.8`，BYOL-gamma `582.4`、RLDP `525.7`、FB `456.2`；DMC-state
`661.2`，FB `648.2`；OGBench-RGB `41.34`，BYOL-gamma `41.58`、FB `39.89`；
OGBench-state `37.98`，HILP `37.98`、FB `39.04`，因此并非每个 suite 都第一。Table 3
（p. 29）symmetric TD-JEPA 在 DMC-RGB 为 `598.1`，完整方法 `628.8`，symmetric
contrastive `437.2`。训练为 DMC 2M 或 OGBench 1M updates，batch 依设置为 512/1024/256，
EMA `.001/.005`；OGBench 还加入 flow behavior-cloning correction（Appendix E，pp. 29-33）。

**优点、局限与本项目关系。** 它已经把 policy-conditioned 长期 latent TD prediction、显式
state/task encoders 和 zero-shot actor 串成强完整系统，是“RL 帮 LeWM”最接近的已发表工作。
但它预测 policy successor features，不是任意 action-sequence dynamics；zero-shot 受 `psi` span
限制，且依赖正交正则、EMA、augmentation 与 BC correction。TDWM 不能再把“TD/RL 帮 JEPA
学习长期控制表征”当贡献本身；可区分空间是保留 LeWM 的 primitive-action counterfactual model，
证明 RL 不破坏 reward-agnostic 物理因子，或把 local LeWM 与 occupancy head 解耦并受控传梯度。

#### Universal Horizon Models（`F`）

**准确题名、作者与发表状态。** *Offline Reinforcement Learning with Universal Horizon Models*，
Hojun Chung、Junseo Lee、Songhwai Oh。所读 PDF 首页标为 **ICML 2026、PMLR 306**，ICML
官方下载页也已收录；[arXiv](https://arxiv.org/abs/2605.15603)、[ICML 官方下载页](https://icml.cc/Downloads/2026)。
截至本次自审未定位独立 PMLR landing page，这一索引项应标“未核实”，不能反过来否定 PDF 的
proceedings 标记。

**故事与模型对象。** `gamma`-model 只能从几何 horizon 混合中采未来，却不告诉使用者样本是
第几步；单步模型则必须递归 rollout。UHM 显式建模
`m_pi(x|s,a,n)=Pr(s_n=x|s_0=s,a_0=a,pi)`，整数 `n` 可任意指定。它仍是“首动作 `a` 后
跟 policy `pi`”的未来分布，不是整段 actions 都由用户指定的 primitive simulator（Sec. 4.1、
Fig. 1，PDF pp. 1-4）。

**训练、损失与价值目标。** Eq. (9-10) 中 `n=1` target 是真实 `s'`；`n>1` 时从
`(s',a'~pi,n-1)` 的 EMA UHM 采未来状态，作为 `(s,a,n)` target。实际使用 coupled
conditional flow matching：复用 Gaussian noise，EMA vector field 经 ODE 生成 bootstrap endpoint，
在线 field 再拟合 transport velocity（Sec. 4.3、Algorithm 1，p. 5）。同时训练 reward model、Q
和 TD3+BC actor；为缓和 offline OOD，下一动作以概率 `beta` 取数据动作，否则取 actor，主实验
`beta=.3`，terminal indicator 拼入 state 并作 absorbing handling。Proposition 4.1 的一般
`nu`-Bellman operator 是 gamma contraction。作者用 winsorized geometric horizon 与 Eq. (15-20)
importance weights 构造合法多步 TD target，并把 lambda 从近 0 排程到终值、用 quantile `q`
决定 `k_max`，避免训练初期依赖不准的大 `n`。

**实验与具体数字。** OGBench 共 100 tasks：50 standard、25 noisy、25 long-horizon。标准/
噪声为 5 seeds、1M updates；long-horizon 为 3 seeds、2M updates/10M transitions。Table 1
（p. 7）standard 平均成功率 UHM `55`、DTD(lambda) `52`、GHM `48`、MBTD `45`、FQL
`44`、MAC `40`；但 AntMaze-large 是 DTD `93` 对 UHM `89`，scene 是 MAC `97` 对 UHM
`43`。noisy 平均 UHM `39`、GHM `38`；long-horizon UHM `22`、GHM `16`、DTD `13`，
而 HumanoidMaze-giant 上 DTD `46`、UHM 仅 `10`（Tables 2-3）。lambda scheduling 和 terminal
handling 都关键，`beta`、`q` 最优值随任务变化。RTX4090 上 long-task UHM update 只比
DTD(lambda) 慢约 10%，明显快于递归 MBTD（Fig. 6，p. 9）。

**优点、局限与本项目关系。** UHM 正面解决“任意显式 horizon `n`”并给 value target fixed-point
理论；但只测 state，依赖 flow、EMA、behavior mixing，在稀疏覆盖和高维 humanoid 上明显失败，
作者也把视觉与 action chunks 留作未来工作。它不是 encoder 防坍塌方法。TDWM 若做显式多尺度
未来，UHM 是必须对比的强 null；LeWM 的差异只能落在 pixel latent、任意 primitive action
sequence、自监督表示或更低计算成本上。

#### Jumpy World Models（`F`）

**准确题名、作者与发表状态。** *Compositional Planning with Jumpy World Models*，Jesse
Farebrother、Matteo Pirotta、Andrea Tirinzoni、Marc G. Bellemare、Alessandro Lazaric、Ahmed
Touati。ICML 2026 [官方下载列表](https://icml.cc/Downloads/2026)已列出题名，但所读版本是
2026-02-23 的 [arXiv v1](https://arxiv.org/abs/2602.19634)，arXiv 元数据未写 venue，独立 PMLR
页面与最终 camera-ready 均未核实。因此严格状态是 **“ICML 2026 官方材料已列入；最终
proceedings/camera-ready 未核实”**，不能无保留地写成已经正式出版或确定 Poster。

**故事与模型对象。** 单个 goal/foundation policy 也许只能完成短技能；作者把多种已训练 policy
当 temporally extended actions，规划“先执行哪个 policy 多久，再切哪个”。Jumpy model 同时条件化
policy `z` 与 timescale/discount `gamma`，从 reward-free offline state data 学 geometric-horizon
successor distribution；测试给新 reward 后不再训练，只组合已有 policies（Sec. 1-3、Fig. 1，
PDF pp. 1-6）。这与 2023 年预测固定长度 action chunks 的另一篇 *Jumpy Models* 不同。

**组合公式、损失与规划。** 每个 policy 以 switching probability `alpha_i` 按几何时长执行，
effective discount `beta_i=gamma(1-alpha_i)`。Theorem 1 分解 switching-policy successor
measure，Lemma 1 给单样本 value estimator。基础模型来自 TD-Flow；Eq. (5) 的 `td-hc` loss
同时对齐真实一步、短 horizon `beta` 的预测、以及短跳后继续 `gamma` 的预测，`beta=gamma`
时退化为普通 TD-Flow。由于 model-on-model target 会放大偏差，只对部分 batch 使用 consistency。
`CompPlan` 用 random shooting 提议 `M` 条 policy/subgoal sequences，逐段采 successor state并按
Lemma 1 估新 reward，只执行首 action/首 policy 后重规划。所有 `alpha=1` 是 action-level MPC；
只选一个持久 policy 是 GPI。AntMaze 用 256 条 goal-conditioned proposals，Cube 用 1024 条
unconditional proposals（Algorithm 2、Appendix p. 21）。

**实验与具体数字。** OGBench AntMaze medium/large/giant 与 cube 1-4；五类 base policies 为
GC-TD3、GC-1S、CRL、GCBC、HFBC。GHM 训练 3M updates、batch 256、3 seeds。Table 1
（p. 7）：HFBC 在 AntMaze-giant 从 `.42` 到 `.79`，GCBC 在 cube-3 从 `.09` 到 `.92`、
cube-4 从 `0` 到 `.76`。Table 2（p. 8）：HFBC+CompPlan 在 giant `.79`，HIQL `.65`、
SHARSA `.56`；cube-4 为 `.67`，对照为 `0/.09`。Fig. 1（p. 9）长任务相对 GPI 提高
`89%`、相对同架构 one-step ActionPlan 提高 `201%`，但这是相对值，部分 base policy 的绝对
成功仍接近 0。

**消融、未核实项与计算边界。** horizon consistency 改善 EMD，例如 giant+GC-1S 从 `7.29`
到 `5.25`，但可规划 horizon 的 success 平均只相对增加约 `5%`（Table 3、Appendix Tables 8-9）。
每 5 步而非每步重规划平均约下降 `20%`；同时优化首 primitive action 比只优化 policy sequence
平均强约 `70%`。计算量为 256/1024 candidates、128/256 evaluation samples、flow ODE 20
steps。原文内部还有冲突：Sec. 4.2（p. 7）写 consistency proportion 为 AntMaze 25%、Cube
12.5%，Appendix Table 11（p. 37）却写 `.25/.15`；未检查代码前 Cube 比例必须标“未核实”。

**优点、局限与本项目关系。** 它已覆盖“跨时间尺度 consistency + 把 RL policies 当抽象动作
规划”，且展示 composition 的实际收益；但只测 state OGBench，switch probability 需手调，
flow planning 昂贵，模型预测已训练 policy 的 occupancy 而非任意 primitive actions，visual latent
仍是未来工作。TDWM 更清楚的定位是双层模型：LeWM 保留局部、视觉、任意动作物理 dynamics；
RL 学到的 jump head只给长期可达性与 proposal。必须比较 ActionPlan、GPI、TD-Flow/Jumpy，并
把 ODE sampling 成本一并报告。

### 14.8 世界动作模型与潜动作

#### DreamZero（`F`）

**主来源与发表状态。** [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
由 Seonghyeon Ye、Yunhao Ge、Kaiyuan Zheng 等人于 2026 年 2 月提交；当前可核实来源是
arXiv v1、作者项目页，以及 [ICLR 2026 World Models workshop 的 OpenReview 条目](https://openreview.net/forum?id=cd33uUB609)。
匿名初稿页眉曾写“under review as a conference paper at ICLR 2026”，但最终公开 venue 是 workshop；
截至 2026 年 8 月未核到 ICLR 主会或期刊的 archival acceptance，作者代码库引用也仍是
`@misc`。因此应写成“workshop 收录 + arXiv 预印本”，而不能写成 ICLR 主会论文。这里的
“zero-shot policy” 指机器人在未见任务或环境上的行为迁移，不是拿到新 reward 后零交互求最优策略。

**论文在讲什么。** 作者认为 VLA 从图文预训练继承了“要做什么”的语义，却没有继承“物体和
机器人将怎样运动”的时空先验。DreamZero 因而不只回归动作，而把一个 14B Wan image-to-video
DiT 改造成 World Action Model：给定视觉历史、语言指令和 proprioception，联合生成未来视频
latent 与同一时间段的 action chunk。论文第 3.1 节的概率分解很重要：联合分布等价于
`future-video prediction × inverse dynamics given that future`。所以视频分支是隐式视觉计划，
动作分支再把该计划翻译成控制；它不是给定一批候选动作、逐一模拟并打分的 MPC world model。

**模型、损失和推理。** 图像经 VAE、文字经 text encoder、机器人状态经 state encoder 后进入
同一个 autoregressive diffusion transformer；多相机画面直接拼成一帧。训练按 chunk 做 teacher
forcing：过去 chunk 保持 clean，当前视频 latent 和归一化动作分别与高斯噪声线性插值；式 (3)
让网络用一个加权 MSE 同时回归两种模态的 flow velocity。初始版本共享视频/动作 denoising time，
DreamZero-Flash 再把视频 time 偏向 noisy 端、动作 time 保持均匀，让动作学会从尚未去净的
视频条件中恢复，以更少视频去噪步换实时性。
梯度确实通过共享 DiT 耦合视频和动作，但没有 reward、value、TD 或 policy-gradient 项。推理时
联合去噪一个 action/video chunk，异步执行动作；动作完成后用真实相机帧覆写 KV cache，丢弃
本轮预测视频，避免长期开环误差。第 3.2 节报告两张 GB200 上约 7Hz，系统、cache、量化和
Flash 合计约 38 倍加速。

**实验读法。** 预训练使用约 500 小时异构机器人数据。AgiBot 的 20 个 seen 与 20 个 unseen
任务各做每机器人 2 次 rollout；seen 平均 task progress 为 `62.2`，最强预训练 VLA 为 `27.4`；
10 个明确 unseen skill 的平均 progress 为 `39.5`，对照为 `16.3`。DROID 未见任务上，DreamZero
为 `49.0% progress / 22.5% success`，GR00T-N1.5 为 `31.0/12.5`，`pi0.5` 为 `33.0/7.5`。
三项 post-training 任务的数据量分别约 33、12、40 小时，每项 10 rollouts；正文 Figure 10 只
给图形结果，不能把 seen-task 的 `62.2/27.4` 误搬成 post-training 平均。表 4 的控制变量更有
解释力：同为 500 小时，diverse data 为 `50`、repetitive data
为 `33`；14B 为 `50`、5B 为 `21`；AR 与 bidirectional 都约 `50`，但 AR 动作更平滑且依靠
KV cache 快 3--4 倍。Flash table-bussing 的 4-step 版本为 `83±6.1`，标准 1-step 为
`52±10.2`，延迟约从 350ms 降到 150ms。附录失败例也显示：视频计划错时，机器人会忠实执行
那个错误计划。

**优点、缺点与 TDWM 关系。** 优点是真机、跨 embodiment、推理系统和数据多样性都有实证，
且把“视频质量与动作质量对齐”做成了可检查接口。限制是每条件试验数很少，task progress 含
人工阶段定义；14B 规模、web-video 初始化、数据多样性和系统优化同时改变，无法把收益单独归因
于 world-model loss。模型只保留约 6 秒上下文、精细接触仍弱，部署成本也远高于普通 VLA。
它最直接支持的是“强自监督视频先验加 action alignment 能改善 policy”，不支持“RL 帮助
LeWM 防坍塌”；若 TDWM 仍做 reward-free candidate-action planning，DreamZero 应列为邻近
WAM 而非等价 baseline。

#### LaWAM（`F`）

**主来源与发表状态。** [LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot
Policies](https://arxiv.org/abs/2606.15768)，Jialei Chen、Kai Wang、Kang Chen、Shuaihang Chen、
Feng Gao、Wenhao Tang、Zhiyuan Li、Weilin Liu、Zhuyu Yao、Boxun Li、Yuanbo Xu、Chao Yu，
2026 年 6 月。当前只有 arXiv v1，未核到正式接收；旧稿若把它写成已发表论文，应改回预印本。

**论文在讲什么。** LaWAM 接受 WAM “先预见未来再出动作”的故事，但认为逐像素视频扩散把
大量算力浪费在纹理上。它先从无 action label 的视频中学习“哪一种潜动作会把当前视觉状态带到
未来”，再把这套潜动作世界模型嵌进语言条件机器人 policy。核心不是 LeWM 式枚举真实动作，
而是从指令直接预测一个 latent action，再把 LaWM 给出的单个 latent subgoal 交给 action expert。

**阶段一：latent action world model。** 冻结 DINOv3 ViT-B/16，得到当前 `u` 与物理时间
`T` 后未来的 `u_T`。posterior `q_phi(z|u,u_T)` 从视觉对推断高斯 latent action；forward decoder
以 `(u,z)` 回归 `u_T`。式 (4) 由三项组成：未来 DINO feature MSE、用 `(EEF_t,z)` 回归
`EEF_T` 的 auxiliary state MSE、以及 `beta KL(q_phi(z|u,u_T)||N(0,I))`。EEF head 只在有机器人
状态的数据上提供物理锚点，训练后丢弃，因此没有 proprioception 也能部署；但这也意味着所谓
“物理 latent action”并非仅由视频目标可辨识。混合频率数据不是固定帧数对齐，而是固定实际
秒数 `tau` 后取目标，附录式 (6)--(7) 给频率编码。

**阶段二：从 latent subgoal 到动作。** policy prior `p_theta(z|o,language)` 蒸馏阶段一 posterior，
其 sampled latent action 经冻结/隔离的 LaWM 一次前向得到未来 feature `u_hat_T`。2.3B policy
的 Alternate-DiT action expert 交替处理 language/scene semantic stream 和 `(u,u_hat_T)` dynamics
stream，以 conditional flow matching 生成 action chunk。总目标包含 posterior-prior distillation、
LaWM future-feature consistency 与 action flow loss。Knowledge Insulation 阻止 action-expert
gradient 回写 LaWM；因此设计上明确优先保护 world model，而不是让控制 loss 改造它。阶段一
LaWM 约 230M 参数，16 张 H100、100k steps、batch 1024；阶段二 64 张 H100、200k steps，
`lambda_distill=lambda_wm=0.1`，全模型 2.3B。

**实验和消融。** 数据约为 3000 小时机器人视频加 1500 小时 egocentric human video。
LIBERO 40 tasks、共 2000 trials：Long/Goal/Object/Spatial 为 `97.0/98.4/99.6/99.4`，平均
`98.6%`，A100 单 chunk 延迟 `187ms`，而 LingBot-VA 报告 `4482ms`。RoboTwin 50 tasks、
每 task 100 trials，LaWAM clean/randomized 为 `92.64/89.80`；必须保留一个反例：它不是
randomized 最好，Fast-WAM 为 `91.98/90.52`，LingBot-VA 为 `91.50/90.92`。三项真机各
30 trials，pick/drawer/towel 为 `93.3/86.7/90.0`，平均 `90.0`，`pi0.5` 平均 `83.3`。
图 5 的消融显示去掉 LaWM 降幅最大，去掉 posterior distillation 也下降，取消 Knowledge
Insulation 并允许控制梯度污染 LaWM 更差；但该图是柱状图，正文未给可逐项核对的精确柱值，
不应伪造数字。附录还说明推理通常 10 个 denoising steps。

**优点、局限与 TDWM 关系。** 优点是把昂贵 pixel WAM 压成一次 latent rollout，并用明确的
gradient barrier 验证“保留预测模型”比端到端覆盖更稳。局限是固定相机前提很强，作者承认相机
运动会被误当 latent action；布料等细粒度变化在 DINO feature 中也容易丢失。LIBERO 已接近
饱和且若干 baseline 数字来自原文、若干为作者复现，比较不是完全同源；human-video 贡献也没有
独立、干净的数量消融。对“RL 帮助 LeWM”而言，这篇反而提供反向证据：最接近 policy 的 loss
被主动挡在 LaWM 外。它是 latent macro-action/WAM 的直接先例，不是 RL 改善 reward-free
LeWM 的先例。

### 14.9 两篇理论工作的可用结论

#### A Generalization Theory for JEPA-Based World Models（`F`）

**主来源与发表状态。** [A Generalization Theory for JEPA-Based World Models](https://arxiv.org/abs/2606.27014)，
Jingyi Cui、Qi Zhang、Hongwei Wen、Yisen Wang，2026 年 6 月；当前是 arXiv v1，未核到正式
接收。论文自称第一篇 JEPA world-model generalization theory，但这个范围限定在其定义的
spectral JEPA，不等同于所有采用 latent prediction 的模型。

**问题和形式化。** 输入是有限状态 `x`、离散 action `a`、下一状态 `x+`；encoder
`f(x) in R^k`，action-conditioned predictor `g(f(x),a)`。作者用 action 条件 co-occurrence
`M(a)` 及其按 state marginal 归一化的 `bar M(a)` 描述真实转移，把正样本内积和负样本平方项
组成 spectral risk。定理 3.1 证明 population risk 等价于
`||bar M(a)-G(F,a)^T F||_F^2 + const`，也就是对每个 action 做 rank-`k` matrix factorization。
下游从 `f(x0)` 连续调用 `g` rollout，选择使终端 latent 与 `f(xg)` 距离最小的 action sequence；
实际评价则看该 sequence 在输入状态转移中到达目标的概率，所以论文真正问的是 latent planner
相对 input-space oracle 的 regret，而非表示是否可解释。

**理论结果逐项读。** 定理 4.1 给单步 bound：regret 不超过
`2 c0 max_a sqrt(R_S-JEPA(f,g,a))`；定理 4.2 在确定 dynamics 和目标边缘分布对 action 不变等
假设下，把多步上界扩为 `2 T c3 sqrt(max_a R)`，明确显示 horizon 线性放大。定理 4.3 用
Eckart--Young 把最优 rank-`k` approximation error 写成 `sum_{i>k} sigma_i(a)^2`。定理 4.4
再用 encoder/predictor hypothesis class 的 Rademacher complexity 控制 finite-sample deviation；
后续定理把两者相加，得到 latent dimension 增大时 spectral tail 下降、但涉及 `k`/`k^2` 的样本
项上升的折中。这里没有 SIGReg、variance regularizer 或 gradient dynamics；风险函数本身含
negative sample 的 spectral contrastive 结构，不能把结论直接套到 LeWM 的 MSE+SIGReg。

**实验到底验证了多少。** 第 5 节只有一个合成 2-D point-mass：真实 state 是
`[p_x,p_y,v_x,v_y]`，observation 再拼 nuisance 与随机噪声；latent 与 input predictive model
用相同 encoder/action encoder/GRU/decoder，规划用 CEM，位置误差小于 `0.08` 算成功。latent
版本用 stop-gradient target MSE 加 variance regularizer，input 版本回归完整 noisy observation。
图 2 表明 1/5 步时两者都接近满分；约 10 步、低噪声时 input model 更强；15--25 步或高噪声时
latent model 更稳，25 步差距最大。论文没有数字表、没有真实图像或机器人实验，图中曲线不应
被转写成虚假小数。实验 loss 也不是理论的 spectral loss，因此它只能说明趋势相容，不是 theorem
的直接数值检验。

**优点、假设和 TDWM 关系。** 优点是把“latent 维度/样本量/规划 horizon”放进一个可审计的
bound，提醒我们降低训练 MSE 并不足够。证据边界也很窄：有限/离散 co-occurrence、bounded
function class、精确优化、确定转移、特殊目标边缘分布和良好 coverage 都很强；现实 partial
observability、连续 action、CEM 近似误差和 policy shift 均未进入理论。它不证明 RL 会改善
LeWM，更不证明 RL 防坍塌；TDWM 可以借它设计 latent-size 与 horizon sweep，但若声称理论保证，
必须重新处理 SIGReg objective、policy-conditioned data 和 stochastic transitions。

##### Metric Non-Collapse 与 JEPA Generalization Theory 的区别

前者从 state-metric hinge 推到 pointwise co-Lipschitz、uniform semiconjugacy 和 optimizer transfer，
要求更强的 state metric、smoothness 与 coverage；后者从 spectral prediction risk 推到 average
planning error，主要刻画 low-rank approximation 与 finite-sample trade-off。二者都表明不能从
训练 MSE 或 global variance 直接跳到“物理世界模型”结论，但所需监督和保证强度完全不同。

### 14.10 LeWM 最新直接邻居

#### Causal-JEPA: Learning World Models through Object-Level Latent Masking（`F`）

**主来源与发表状态。** [Causal-JEPA: Learning World Models through Object-Level Latent Masking](https://arxiv.org/abs/2602.11389)，
Heejeong Nam、Quentin Le Lidec、Lucas Maes、Yann LeCun、Randall Balestriero。arXiv v2 的
comments、论文首页的 `PMLR 306` proceedings 标记以及 [ICML 2026 官方 poster 页](https://icml.cc/virtual/2026/poster/63623)
相互印证，故可写作 **ICML 2026 正式发表**；不能只把 2026 预印本日期当发表证据。

**故事与数据流。** 普通 object-centric predictor 很容易只看目标对象自己的连续轨迹，学成
惯性外推而不理解碰撞或动作影响。C-JEPA 因而把 object slot 的“可见性”做干预：每帧由冻结
VideoSAUR 聚合冻结 DINOv2 features 得到 6 个 object slots，另以 SAVi 做可比实验；action 和
proprioception 是独立 auxiliary entity tokens。训练时随机选对象，将它们在整个 history 的 slot
遮掉，只留下最早帧经线性映射的 identity anchor；所有 future entities 都遮掉。双向 ViT 因为看
不到被选对象的自轨迹，只能借其他对象和 action/proprio 补全它，并同时预测未来。推理时 history
完全可见，仅 future 置 mask，所以没有训练期遮挡造成的控制输入缺失。

**损失、梯度与规划。** 训练只在被 mask 的 token 上做 latent MSE，式 (6) 分成 history completion
与 future prediction 两部分：

```math
\mathcal L_{mask}=\mathbb E\!\left[\sum_{\tau,i}
\mathbf 1[\bar z_\tau^i\ne z_\tau^i]\|f(\bar Z)_\tau^i-z_\tau^i\|_2^2\right]
=\mathcal L_{history}+\mathcal L_{future}.
```

梯度更新 predictor 和其 auxiliary encoders，不更新冻结 object/DINO backbone，也没有 reward、
value 或 RL loss。Push-T 用 Hungarian matching 对齐 permutation-equivariant slots，终端 object
latent 到 goal 的 MSE 作为 CEM cost。附录 G 的具体设置是 3-frame history、未来 5 个 block、
每 block 5 actions，总计 25 primitive actions；300 candidates、30 elites、30 iterations，执行整段
25 actions 后才重规划，最多 50 env steps。

**实验和消融。** CLEVRER 是 10k/5k/5k videos、每段 128 frames；VQA 因服务器不可用而在
validation split 按 test protocol 报告。表 2 中 VideoSAUR 的 OC-JEPA 无 history mask 为
`82.79%` 平均、counterfactual per-question `47.68%`；mask 4/7 objects 后为 `89.40/68.81`，
后者提升 `21.13` 点。SAVi 则 mask 2 个最佳 `83.88/60.19`，mask 4 个反而跌到
`73.28/34.06`，说明不是遮得越多越好。Push-T 表 6：patch DINO-WM `91.33%`、
DINO-WM-Reg `88.00%`、OC-DINO-WM `60.67%`、同架构 OC-JEPA `76.00%`、C-JEPA
`88.67%`；6×128 slots 相对 196×384 patch features 只有约 `1.02%` 输入量。每模型是 50 个
trajectory pairs、3 seeds；论文报告 L40S 上规划约 `673s`，DINO-WM 约 `5763s`。附录 A10 的
mask 对照同样关键：object mask 1/4 为 `88.67`，token mask `84.67`，tube mask `55.33`；
object mask 2/4 为 `82.67`，tube mask降到 `5.33`。

**优点、严格边界与 TDWM 关系。** 优点是有同架构 OC-JEPA 控制变量、mask 粒度负对照和真实
planning efficiency，而不只是展示 attention 图。论文也诚实限定“causal”：定理只说 conditional-
expectation optimum 必须使用 masked completion 的最小 predictive influence neighborhood；该集合
可以包含共同混杂造成的相关变量或因果后果，并非 causal parents。理论另假设 object-aligned sufficient
slots、shared stationary mechanism 和 finite-history sufficiency。性能依赖预训练 slot identity 与
Hungarian alignment，控制只测 Push-T；patch 与 slot 模型的 predictor/训练轮数也不完全相同。
它直接覆盖“对象级遮挡让 LeWM 更关注交互”的方向，却没有 RL、TD、occupancy 或 anti-collapse
新机制。对 TDWM，它更适合作为结构先验正交轴，而不是“RL 帮 LeWM”的先例。

#### Hierarchical Planning with Latent World Models / HWM（`F`）

**主来源与发表状态。** [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208)，
Wancong Zhang、Basile Terver、Artem Zholus、Soham Chitnis、Harsh Sutaria、Mido Assran、
Randall Balestriero、Amir Bar、Adrien Bardes、Yann LeCun、Nicolas Ballas，arXiv v2，2026 年
6 月。当前主来源只证明预印本；“under CoRL review”即使作者主页曾写过也不是接收状态，旧稿
不能写成已发表或已接收。

**故事和两层算子。** flat latent WM 在长任务上同时遇到 rollout error 累积与 action search 指数
增长；作者认为失败未必是 representation 差，而可能只是缺少中间 subgoal。HWM 保留同一 frozen/
shared observation latent，低层 `F1(z_t,a_t)` 学短尺度转移，Transformer action encoder `A_psi`
把一段 primitive action chunk 压成 macro-action `l_t`，高层 `F2(z_t,l_t)` 直接预测更稀疏的
waypoint latent。低层既有 teacher-forced L1，也有多步 autoregressive rollout loss；高层用真实
chunk 编码和真实 waypoint 做 teacher-forced L1。macro-action 没有 reward、skill-discovery 或
policy-gradient supervision，而是纯 action-sequence compression。

**规划信息流。** 高层先以 CEM/MPPI 搜 macro-action sequence，使最后的 high-level prediction
接近最终 goal；第一段预测 waypoint 变成低层 subgoal。低层再搜 primitive actions 到该 subgoal，
执行短段后两层一起重规划。因此 high model 降 rollout 次数，low model 保局部精度。Franka 用
4-D macro action、3 个 waypoints、最长 4 秒片段；Push-T 接 DINO-WM，maze 接 PLDM。它是
planner/model interface 的贡献，不是某一个 encoder loss。第 4.2 节实测 2 秒预测时 low model
要自回归最多 16 步，高 model 一步；1 秒以内 low 更准，1.5 秒以后 high 更准。

**具体实验与消融。** Franka 约 130 小时 DROID+RoboSet，输入 RGB+EEF pose；cup/box 分别
10 个 start-goal pairs×5 trials，drawer 7×5。表 1 flat VJEPA2-AC 为 `0/0/30%`，HWM 为
`70/60/70%`；`pi0-FAST` 为 `52/18/-`，`pi0.5` 为 `68/36/-`，但后两者用语言 goal，接口不
同。人工 intermediate subgoal 可让 flat planner 达到 cup/box `80/80%`，而 end-to-end 为
`0/0`，这是论文最有力的“planner 而非 model”归因。Push-T 在距离 25/50/75 时，flat DINO-WM
`84/55/17%`，hierarchy `89/78/61%`；matched-capacity flat 98M 在 H=15 仅 `15%`，94M
hierarchy `61%`。20 个 unseen 10×10 mazes 上，easy/medium/hard：PLDM `100/84/44%`，HWM
`100/95/83%`，HIQL `88/73/48%`。正文图 5 总结为最高约 3 倍 test-time compute 优势，maze
文字另报告 matched success 可达约 4 倍，二者口径不要混写。高层 stride 6/8/10/12/14 的 hard
maze 为 `50/55/77/78/75%`；macro dimension 太低不能表达计划，太高则 subgoal 可达性下降，
Franka 以 4 维最好。

**优点、严格边界与 TDWM 关系。** 优点是 manual-subgoal oracle 与 matched-parameter ablation
真正隔离了 hierarchy 收益，并跨三种 backbone。局限是 sampled macro-action 未必来自数据支持，
也没有 decoder 保证能还原 primitive chunk；高层终点正确不代表第一个 waypoint 可控，作者正是
用 expert-action cosine 才发现这个问题。粗 subgoal 还会丢深度/接触精度，真机 trial 数也很小。
它已覆盖“多时间尺度 latent WM + macro-action + hierarchical MPC”，但没有用 RL 改善 LeWM；
若 TDWM 提案只是再加高层/低层，就与 HWM 重合，必须把 novelty 放在 policy-conditioned TD
target、coverage 或可达性约束上。

#### Subspace-Decomposed JEPAs / SD-JEPA（`F`）

**主来源与发表状态。** [Subspace-Decomposed JEPAs: Disentangling Progression and Content in
Latent World Models](https://arxiv.org/abs/2605.31111)，Lucas Thil、Jesse Read、Rim Kaddah、
Guillaume Doquet，2026 年 5 月；当前为 arXiv v1，没有正式接收记录。

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

**训练和梯度边界。** 完整目标是 full-latent one-step prediction、content-only SIGReg、progression-
only temporal cosine-margin triplet、以及 consecutive latent velocity 的 straightening。`theta,r` 与
action 一起 condition predictor；规划 cost 是 content MSE 加可选 angular/radial goal cost，仍由
CEM 搜动作。论文 Proposition 1--2 只能保证 triplet 与 SIGReg 对 latent coordinates 的直接梯度
support 正交；正文明确承认回到共享 encoder 参数后两种梯度仍会相加，不能宣称消除了所有 gradient
conflict。角度只在全局旋转下可辨识，也不等于物理时间。

**实验与数字。** 完全沿用 LeWM 四环境：50-step evaluation、goal offset 25、planning horizon 5、
frame skip 5；Push-T CEM 为 300 candidates/30 iterations，其余 10 iterations。ViT-tiny 加 6-layer
predictor 约 18.04M，10 epochs、batch 128、seeds `{0,42,3072}`。表 1 的正确数字是：LeWM
Two-Room/Reacher/Push-T/Cube `87/86/96/74`；按环境挑最佳 `k` 的 SD-JEPA 为
`90/88/97.3/72`，即 `+3/+2/+1.3/-2`。必须更正正文 5.4 节的一处内部不一致：文字声称 Reacher
`k=4` 为 `92`，但主表 1 和表 3 的 3-seed mean 是 `88`；正式报告以表值为准并标明冲突，不能
悄悄写 92。表 3 的 `k=2/4/8` 分别为 Push-T `94/96/97.3`、Two-Room `90/88/90`、Reacher
`84/88/83.3`、Cube `72/69.3/69.3`，说明需要逐环境调 `k`。

**消融与表征诊断。** Push-T 单 seed falsifier：baseline 96；取消 split、在 full latent 加 triplet
仍为 96；正确 split 为 98；split 后把 triplet 错施到 full latent 降到 92。这支持 coordinate split，
但单 seed 证据不能当稳定效应。progression-only goal cost 仅 `28%`，说明低维“进度”不足以独立
表示目标。Cube 40 episodes、160 contact events、1480 steps 的 AUROC：容差 ±1/2/3 时 z-MSE
为 `.238/.360/.513`，`|Delta theta|` 为 `.414/.473/.565`；但最紧容差二者都低于 .5，正确读法
是相对 gap。8-D progression 只占 latent `4.2%`，within-episode linear probe R2 在 Cube/Push-T/
Reacher/Two-Room 为约 `.905/.908/.948/.717`；pooled cross-episode probe 多为负，故它是每条
trajectory 的 phase coordinate，不是全局可校准距离。

**优点、局限与 TDWM 关系。** 优点是它真的对准 LeWM、参数量和训练预算匹配，并做了错误
target 的反事实消融。局限是最佳 `k` 依环境选择、固定坐标 split 与圆周几何先验很强，四环境中
Cube 退步，主要因果消融还是单 seed；其 “physical progression” 往往与 step clock 同样相关，
跨 episode 又不校准。它直接封住“把 LeWM latent 切一块做 temporal triplet/phase”这一宽泛
想法，但没有 RL、successor distribution 或 policy occupancy。TDWM 必须说明 TD/RL signal
提供了什么超出这个显式 progression subspace 的信息。

### 14.11 经典基于模型强化学习与想象训练

#### World Models（`F`）

**主来源与发表状态。** [World Models](https://arxiv.org/abs/1803.10122)，David Ha 与 Jürgen
Schmidhuber，2018，是原始长篇交互版；其压缩会议版改题为
[*Recurrent World Models Facilitate Policy Evolution*](https://proceedings.neurips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)，
正式发表于 **NeurIPS 2018 主会**（Advances in Neural Information Processing Systems 31）。官方
proceedings 也明确把 `worldmodels.github.io` 称为该论文的 interactive version。因此旧报告若说
它“只有 arXiv、没有 NeurIPS 发表记录”是错误的；严谨写法要同时区分长篇标题与 proceedings 标题。

**故事、模型与数据流。** 论文把复杂度拆成 Vision、Memory、Controller。VAE 逐帧把 RGB 压成
32-D `z_t`，以 reconstruction+KL 学空间表征；MDN-RNN/LSTM 输入 `(z_t,a_t,h_t)`，最大化
下一 `z` 在 Gaussian mixture 下的 likelihood，从而表达多模态未来；线性 controller
`a_t=W[z_t,h_t]+b` 只含 867 个参数，用 episode return 的 CMA-ES 黑盒优化。三部分分阶段训练，
controller 没有反向更新 VAE 或 RNN。CarRacing 从随机 policy 收集 10,000 rollouts，VAE 约
4.35M 参数、MDN-RNN 422k；world model 完全看不到 reward，只有 controller 看 reward。

**控制和“dream”究竟做了什么。** CarRacing 的 controller 在真实环境内优化，只把 learned
`z,h` 当 feature；VizDoom 才把 MDN-RNN 包成虚拟 Gym 环境，在其中用 CMA-ES 训练 controller，
再零微调移回真实游戏。temperature `tau` 增大 mixture sampling 的随机性，刻意把 dream 变难，
以免 controller 钻 deterministic simulator 漏洞。第 4.5 节的“cheating”是重要负结果：低温模型
里 policy 可找到 model 不真实的生存策略，dream score 很高而真实失败；提高随机性相当于鲁棒化，
不是提高预测 fidelity。

**实验数字。** CarRacing 100 个随机 tracks 上，只用 VAE latent 的线性 controller 为
`632±251`，加 hidden layer 为 `788±141`，完整 `[z,h]` 线性 controller 为 `906±21`；当时
leaderboard 为 `838±11`，A3C discrete `652±10`。VizDoom Take Cover 以平均存活超过 750 steps
算 solved，dream 训练的 controller 回真实环境约能达到 1,100 steps 量级，但论文的核心证据是
温度与 transfer 曲线，不应把单次曲线读成精确统一 benchmark。作者还在第 5 节提出可迭代用新
controller 收集数据再重训 M，但主实验不是端到端 RL-to-model。

**优点、局限与 TDWM 关系。** 优点是很早就分离了 predictive representation 与 task controller，
并直面 model exploitation。局限是小型视觉任务、随机数据覆盖、pixel reconstruction 和独立训练；
没有 uncertainty calibration、goal-image MPC 或多任务 revaluation。它证明“policy 可以在 world
model 中学”，不证明“RL loss 让 world model 更物理/防坍塌”。对 TDWM 更重要的遗产是：只要
RL 开始查询模型，就必须用真实 rollout、uncertainty 或 conservative constraint 检查利用误差。

#### PILCO（`F`）

**主来源与发表状态。** [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://icml.cc/2011/papers/323_icmlpaper.pdf)，
Marc Peter Deisenroth、Carl Edward Rasmussen，ICML 2011，正式发表（pp. 465--472）。

**故事与模型。** PILCO 的目标不是深度 latent，而是极少真实试验下怎样避免模型过度自信。对
低维连续状态增量学一组独立 Gaussian Processes：`x_{t+1}=x_t+f(x_t,u_t)+epsilon`，每个输出
维度用 SE-ARD kernel，hyperparameters 由 marginal likelihood 定。因为未来 state 不是点而是
分布，当前 Gaussian state 经确定 policy 形成 uncertain GP input；论文用解析 moment matching
求 GP 输出和 state-input 交叉协方差，再逐步把整个 horizon 近似为 Gaussian。与只 rollout GP
均值不同，epistemic/model uncertainty 会进入预期 cost。

**损失、梯度与循环。** policy 目标是 `J(theta)=sum_t E[c(x_t)]`，实验用饱和的非二次 cost，
远离目标时梯度不会像平方损失一样爆炸。GP posterior moments、moment propagation、expected
cost 和 RBF/线性 policy 全部可微，论文逐层求 `dJ/dtheta`，用共轭梯度或 L-BFGS 找 policy；
执行一条真实 trial 后把 transitions 加回集合、重新拟合 GP，再优化 policy。RL cost 的 gradient
穿过模型去更新 policy，但 GP 参数仍由 transition likelihood 学，不由 return/value 改写。

**实验数字。** 真机单摆/小车摆起用 4-D state 和 50 basis-function policy（305 参数），总共约
`17.5s` 真实系统经验即可摆起并平衡；真机双摆用 6-D state、1816 参数，约 20--30 trials、
`60--90s` 经验；仿真 5-DoF unicycle 是 12-D state、28 参数 policy，约 20 trials、20--30 秒
经验，1000 次测试约 `93%` 成功。这些数字展示 sample efficiency，但任务均低维、全状态可见，
不能与 pixel world model 的环境步数直接横比。

**优点、局限与 TDWM 关系。** 优点是 uncertainty propagation 与 policy gradient 数学闭环清楚，
每次真实试验都由当前模型主动选择。局限是 GP `O(n^3)`、每步 Gaussian moment approximation、
smooth dynamics、可微 policy/cost 和全状态假设，不适合直接扩到图像与长 horizon。PILCO 支持
“RL 通过选择高价值/高不确定数据间接帮助模型覆盖”，不支持“return gradient 是 encoder
anti-collapse loss”。TDWM 可以借其 uncertainty-aware weighting 思想，但需在 LeWM latent 上
重新验证 calibration。

#### PETS（`F`）

**主来源与发表状态。** [Deep Reinforcement Learning in a Handful of Trials using Probabilistic
Dynamics Models](https://proceedings.neurips.cc/paper_files/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)，
Kurtland Chua、Roberto Calandra、Rowan McAllister、Sergey Levine，NeurIPS 2018 正式发表。

**故事和不确定性分解。** 作者认为 learned model-based control 的瓶颈不是网络容量，而是不能
区分环境本身的随机性与“没见过所以不知道”。PETS 用默认 5 个 bootstrap probabilistic neural
networks，各自输出 next-state delta 的 diagonal Gaussian，Gaussian NLL 同时学 mean 与
aleatoric variance；不同 bootstrap members 的分歧代表 epistemic uncertainty。控制时每条 candidate
action sequence 展成约 20 particles。`TS1` 每步给 particle 重采 ensemble member，近似每步
随机模型；`TS∞` 让同一 particle 整段固定一个 member，保留“哪一个 dynamics 假说”造成的
轨迹相关性。模型没有视觉 encoder，输入是真实 state。

**算法和梯度。** CEM 对 candidate action distribution 反复 sample、按 particles 的已知 reward
求期望回报、用 elites 重拟合，只执行第一动作；收集真实 transition 后把它加入数据，每 trial
重新训练 ensemble。return 不反向进入 dynamics NLL；RL 只决定 action query 和下一批真实数据。
论文还尝试直接通过 stochastic model 对 policy 做 gradient，发现梯度混乱而失败，称为 chaotic
policy gradients，这一点正好警告“让 RL loss 直接穿过不确定 model”未必稳定。

**实验与消融。** 任务为 cart-pole、7-DoF pusher/reacher、HalfCheetah；预算小于 100k time
steps/约 100 trials，通常 10 runs，对比 PPO、DDPG、SAC、MBMF 和 GP 方法。摘要给出的代表性
sample-efficiency 是 HalfCheetah 相对 SAC 约 8 倍、相对 PPO 约 125 倍更少数据达到相近表现。
消融表明 probabilistic ensemble 几乎总是最好，只有简单 cart-pole 例外；模型表示方式的影响大于
particle propagation 变体，GP moment matching 在高维 state 上已难扩展。结果图多于数字表，
不应从曲线伪造最终 return 小数。

**优点、局限与 TDWM 关系。** 优点是把 epistemic/aleatoric 和 trajectory propagation 拆得清楚，
也是强健 MPC baseline。限制是 simulator state、known reward、短 MPC、昂贵 CEM，没有 representation
collapse、pixel distractor 或 cross-task transfer。它支持 RL-driven data acquisition，不能证明
value loss 让 LeWM latent 更物理；TDWM 若主动采数，PETS ensemble uncertainty 是必须比较的
简单解释。

#### MBPO（`F`）

**主来源与发表状态。** [When to Trust Your Model: Model-Based Policy Optimization](https://proceedings.neurips.cc/paper_files/paper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf)，
Michael Janner、Justin Fu、Marvin Zhang、Sergey Levine，NeurIPS 2019 正式发表。

**故事与理论。** 长 model rollout 能省真实数据，却把每步 model bias 连乘；MBPO 不追求“先把
model 学到足够完美”，而从真实 replay state 分叉，只生成短 synthetic branches。理论先给 policy
在模型和真实 MDP 下 return 差的 bound：它随 rollout length `k` 和 one-step model error 线性增大；
严格按 worst-case bound 最优甚至会退化到 `k=0`。作者因此没有把 theorem 当性能证明，而用
empirical model generalization 决定训练中逐步增加短 horizon。

**算法细节。** probabilistic ensemble 从真实 buffer `D_env` 最大似然预测 next state 和 reward。
每轮先在真实环境收集 `E` steps；再从 `D_env` 均匀抽真实 states，使用当前 SAC policy 在随机
ensemble member 中 rollout `k` 步，把 transitions 放入短寿命 `D_model`；SAC 从两个 buffers
混采并做每真实 step 约 20--40 个 gradient updates。典型 schedule 从 `k=1` 缓慢增到 15，但
核心不是 15 步，而是大量、分散的短分支。policy gradient 不进入 model loss，model 数据分布则
随当前 policy 改变。

**实验与消融。** 标准 1000-step MuJoCo 对比 SAC/PPO、PETS、STEVE、SLBO，通常 5 trials。
Ant 上 MBPO 约 300k env steps 匹配 SAC 约 3M；作者按 wall-clock 报告 Hopper、Walker2d 达
等效表现约 14/40 分钟 simulator interaction。图 3 最重要：单纯提高 model-free SAC update ratio
不能解释收益；Hopper 上 `1→15` schedule 最好，但固定 `k=1` 仍很强；模型在约 200-step rollout
的 state-error 图上看似准确，训练 policy 时使用长 rollout 却更差，500 步尤其明显。单步 MVE
value expansion 也强，而与 MBPO 组合未再增益。短 model return 与真实 return 高相关，且模型
往往保守低估。

**优点、局限与 TDWM 关系。** 优点是用 branched rollout 把 model bias 变成可调 knob，并用
high-update SAC 消融排除“只是多算梯度”。局限是 state-based MuJoCo、task reward 已知、没有
视觉表征或 zero-shot revaluation；theory bound 保守到不能推荐 `k>0`。MBPO 明确属于“RL 消费
world model 并改变其访问分布”，不是“RL loss 训练 world model”。它对 TDWM 的主要约束是：
即便 LeWM 多步 MSE 好看，也要做 policy-induced rollout / exploitation 测试，并优先比较短 TD
branch，而不是假定越长越物理。

#### Plan2Explore（`F`）

**主来源与发表状态。** [Planning to Explore via Self-Supervised World Models](https://proceedings.mlr.press/v119/sekar20a.html)，
Ramanan Sekar、Oleh Rybkin、Kostas Daniilidis、Pieter Abbeel、Danijar Hafner、Deepak Pathak，
ICML 2020，PMLR 119:8583--8592，正式发表。

**故事与模型。** 一般 curiosity 奖励“刚才没预测对”，容易追逐 aleatoric noise；Plan2Explore
要在行动前规划未来 epistemic novelty。全局 world model 是 PlaNet/Dreamer RSSM：image encoder、
posterior、action-conditioned prior、image decoder 和 dynamics，以无外在 reward 的 ELBO 学。
旁边放 5 个两层 MLP，每个从 `(model state, action)` 回归下一时刻 1024-D encoder embedding，
用 MSE 独立训练。它们 predictive means 的方差是 intrinsic reward。论文式 (8)--(9) 固定每个
ensemble member 的 conditional variance，因此 disagreement 只近似“关于 mean 的 information
gain”，不是完整 Bayesian posterior uncertainty。

**RL 怎样进入。** Dreamer actor/value 完全在 RSSM imagination 中最大化未来一串 disagreement，
比 one-step retrospective surprise 更 farsighted；该 policy 回真实环境收集新 episode，RSSM 和
ensemble 再做自监督更新。下游测试时，用户仍要提供 reward function；系统先用已有 replay
给所有 transition relabel reward、训练 reward predictor，再在冻结的 task-agnostic model 内训练
新 task policy。所谓 zero-shot 指没有新的环境 interaction，不是“不需要 reward / 不再训练”。
RL gradient 更新探索 actor/value，不直接更新 RSSM encoder；模型改善来自它采到更有信息的
真实数据。

**实验数字和消融。** DMControl 64×64 images、episode 1000、action repeat 2，通常 3 seeds；
main text 展示 8 个任务，supplement 给全 20 个。3.5M reward-free steps 后的 zero-shot 平均分：
Plan2Explore `563.58`、Curiosity `489.26`、Random `342.11`，从头用 task reward 训练的 Dreamer
oracle `694.77`。例如 Cheetah Run 为 `784.45` 对 `495.55/0.78/888.84`；但 Quadruped Walk
是反例，Plan2Explore `182.87`，Curiosity `368.45`，Dreamer `921.25`。1M unsupervised 加
150k task-specific adaptation 的全任务平均为 `643.73`，Curiosity `537.95`、Random `155.23`、
MAX `210.46`、Retrospective `258.78`、Dreamer oracle `700.27`。未来 disagreement 优于只最大化
当前/过去 novelty；intrinsic scale 固定乘 `10,000` 且未 normalization，是一个敏感超参数。

**优点、局限与 TDWM 关系。** 这是“RL 帮 world model”最直接、最干净的正例之一，但帮助
通道是 active data acquisition。优点是 reward-free exploration 与多任务 adaptation 分开，且有
20-task supplement。局限是仍需在线真实交互、ensemble disagreement 会受 function initialization/
noisy-TV 影响、只测试同一环境内换任务、3 seeds 且 3.5M steps 并不小。对固定 offline LeWM，
不能照搬采数收益；TDWM 若主张 RL 直接塑造 representation，必须与 Plan2Explore 的“只换数据、
不回传 RL 到 encoder”对照，否则最简单解释就是 coverage。

#### PlaNet（`F`）

**主来源与发表状态。** [Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a.html)，
Danijar Hafner、Timothy Lillicrap、Ian Fischer、Ruben Villegas、David Ha、Honglak Lee、James
Davidson，ICML 2019，PMLR 97，正式发表。算法名 PlaNet（Deep Planning Network）。

**故事与 RSSM。** PlaNet 想从 64×64 pixels 学一个足够快、足够随机、又有记忆的 latent simulator，
让 CEM 每个真实 step 评估上千 action sequences。RSSM 把 state 分成 deterministic `h_t` 与
stochastic Gaussian `s_t`：GRU 用前一 `(h,s,a)` 更新记忆，prior `p(s_t|h_t)` 在没有新图像时
rollout；filter posterior `q(s_t|h_t,o_t)` 看图纠正 belief；image decoder 和 reward head 从
`(h,s)` 预测 observation/reward。序列 ELBO 是 image/reward likelihood 减 posterior-prior KL，
reparameterized sample 端到端更新。所有观察信息必须经过 stochastic bottleneck，防止 decoder
绕开 dynamics。

**规划和数据循环。** CEM 每步从 diagonal Gaussian 采 `J=1000` 条 horizon `H=12` action
sequences，做 `I=10` 轮、保留 `K=100` elites；每 candidate 只采一条 latent trajectory，累加
predicted mean rewards，不解码图像；执行第一动作后重新从 zero-mean/unit-variance action belief
开始。5 个 random seed episodes 后，每轮模型训练 100 updates 再收一 episode，并加 Gaussian
action noise。没有 actor/value network；RL 的 reward 训练 reward head并决定在线采样，但没有
独立 policy-gradient loss。

**必须更正 latent overshooting 的旧叙述。** 第 4 节确实推导多距离 prior 对未来 posterior 的 KL，
`d>1` 时 stop-gradient posterior target，用来训练 multi-step transition。可是最终 RSSM agent **没有
使用它**：supplement Appendix A 明说 previous version 才使用，Appendix D Figure 8 显示它显著
帮助 DRNN，却“slightly reduces” RSSM performance。旧报告若把 overshooting 写成最终 PlaNet
成功的必要模块是错误的；正确说法是论文提出并分析它，但最终 RSSM 不需要。

**实验数字与消融。** 六个 DMControl pixel tasks、1000 episodes、5 seeds、每 seed 10 evaluation
trajectories。表 1 PlaNet 在 cartpole/reacher/cheetah/finger/cup/walker 为
`821/832/662/700/930/951`；A3C proprio 100k episodes 为 `558/285/214/129/105/311`，D4PG
pixels 100k 为 `862/967/524/985/980/968`，true-simulator CEM 为 `850/964/656/825/993/994`。
论文估算相对 D4PG 达同分的数据效率为 `250/40/500+/300/100/90` 倍。图 4 显示纯 GRU 与纯
stochastic SSM 都差，尤其无 stochastic path 时多任务不学习；图 5 表明 online planned collection
在 cartpole/finger/walker 必需，CEM 也全面优于 random shooting。单 V100 训练约 10--20 小时。

**优点、局限与 TDWM 关系。** 优点是把 filtering、stochastic rollout、reward prediction 和 MPC
做成完整 pixel agent，并用 architecture/data/planner 三组消融。局限是 task reward 进入 model、
单任务在线采数、CEM 每步昂贵，generation fidelity 与 control return耦合；实验也不检验 frozen
model 换 reward。它支持“planning policy 改变数据分布”而不是“RL gradient 防表示坍塌”。对
TDWM 最重要的两个教训是 stochastic/deterministic memory 都要做对照，以及不能从 multi-step
auxiliary loss 的提出直接推断它在最终模型有效。

#### Dreamer（`F`）

**主来源与发表状态。** [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)，
Danijar Hafner、Timothy Lillicrap、Jimmy Ba、Mohammad Norouzi，ICLR 2020 正式发表。

**故事与模型数据流。** PlaNet 每个真实 step 做 CEM，且固定 horizon 内只看即时 model reward，
既慢又可能短视。Dreamer 保留 RSSM：真实 replay 经 posterior 得到 model states，prior 从
`(state,action)` 在无图像时前推，observation/reward heads 用 sequential ELBO 学；再把 search
amortize 成 actor，并用 critic 概括 imagination horizon 之后的回报。从 replay 的每个 posterior
state 同时起 imagined trajectory，所以一批真实序列能产生大量并行训练状态。

**损失、梯度和推理。** world-model reconstruction objective 是 observation log-likelihood、reward
log-likelihood 加 posterior/prior KL。actor 输出 tanh Gaussian，以 reparameterization sample action；
critic 拟合 `lambda=0.95` 的多步 return，默认 imagination horizon 15、discount .99。actor 最大化
沿 imagined trajectory 的 lambda-return，gradient 经过 action sample、latent transition、reward
与 value predictions 回到 actor。式 (7)--(8) 旁的原文明确写着：**world model is fixed while
learning behaviors**。因此“gradient through dynamics”是用模型的 Jacobian 求 actor gradient，
不是 actor loss 更新 dynamics。部署时 posterior filter 当前 history，一次 actor forward 出动作，
不再 CEM。

**实验与关键消融。** 20 个 DMControl visual tasks、64×64、通常 action repeat 2、5 seeds；5 个
random seed episodes 后每收一 episode 做 100 updates，batch 50、sequence 50。5M env steps
平均分 Dreamer `823`，PlaNet `333`，D4PG 在 100M steps 为 `786`；单 V100 每 1M steps 约
3 小时，PlaNet 约 11 小时，D4PG 达相近表现约 24 小时。Figure 4/7 显示无 critic、只最大化
horizon 内 reward 的 actor 与 online planner 在 acrobot/hopper 等长任务短视，value bootstrap
使 horizon 变化更稳。表示学习消融比较 pixel reconstruction、contrastive InfoNCE 与 reward-only：
reconstruction 在多数任务最强，contrastive 约能解决一半，稀疏 reward-only 不足；这比“RL 自然
学出物理 latent”更接近论文真实结论。45-step decoded prediction 只是定性证据，不应等同于
long-horizon counterfactual correctness。

**优点、局限与 TDWM 关系。** 优点是把 analytic imagination gradient、lambda-return 与高吞吐
latent rollout 组合成熟，并明确比较 representation objectives。局限是 task reward head、在线 replay
和 reconstruction 绑定；world-model accuracy 多由最终 control 间接评价，actor 仍会利用 model
bias。它证明 RL 能高效利用 world model，不能证明 RL loss 改善 world model。对“RL 帮 LeWM”，
Dreamer 给出的更稳妥路线是冻结 LeWM 做 actor/value consumer，并单独测 RL-guided data collection；
若让 actor gradient 回写 encoder，那是超出 Dreamer 的新实验，必须加 stop-gradient 对照。

#### DreamerV2（`F`）

**主来源与发表状态。** [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)，
Danijar Hafner、Timothy Lillicrap、Mohammad Norouzi、Jimmy Ba，ICLR 2021 正式发表。

**故事和模型变化。** V2 要把 Dreamer 从连续控制扩到 Atari 55 games。RSSM 仍有 GRU
deterministic state，但 stochastic state 改成 32 个 categorical variables、每个 32 类，总计 1024
维 sparse one-hot、每时刻仅 32 个 active；posterior 看 image，prior 不看 image，decoder/reward/
discount heads 从 latent 预测。categorical sample 用 straight-through estimator。这不是为了可解释，
而是让 model 更容易表达离散、多模态游戏状态。

**world-model 与 behavior loss。** model loss 是 image、reward、discount negative log-likelihood
加 `beta KL(posterior||prior)`；Atari `beta=0.1`。KL balancing 用约 `alpha=0.8`：一项 stop-gradient
posterior、主要训练 prior 追 posterior；另一项 stop-gradient prior、较慢训练 representation 追 prior。
行为学习从 replay posterior states 想象 horizon 15，critic 拟合 lambda-return 并有每 100 steps
更新的 target critic。Atari actor 为 categorical policy，混合 REINFORCE 与 straight-through
dynamics gradient；Atari 配置 `rho=1`，即纯 REINFORCE，entropy scale `1e-3`。一次可并行约
2500 imagined trajectories。和 V1 一样，actor/value 更新时 world model 固定。

**实验数字与消融。** Atari 55、sticky actions、200M environment frames（action repeat 4）、
full action space、no life information，每 game 单独 agent；单 V100 少于 10 天。表 1 human-
normalized gamer median/mean 为 `2.15/11.33`，record-normalized mean/clipped mean为 `.44/.28`；
IQN 对应 `1.29/8.85/.21/.21`，Rainbow `1.47/9.12/.17/.17`。早期版本的 clipped Atari
ablation 表 2：full `.25`，去 discrete `.19`，去 KL balance `.16`，去 policy reinforce `.15`，
stop image gradients 几乎归零到 `.01`，stop reward gradients 仍 `.24`。逐 game 看 discrete latent
42 胜/8 负/5 平，KL balance 44/6/5；image gradient 在 51 个 games 有害地被移除，而 reward
gradient 只在 22 个有益、15 个反而变好、18 个持平。

**优点、局限与 TDWM 关系。** 优点是大规模逐游戏消融直接说明 world-model 表征主要靠 dense
image self-supervision，而非稀疏 reward；KL 两侧 stop-gradient 也提供了稳定 prior/representation
接口。局限是每 game 单模型、200M frames 和约 468B imagined states 的巨大 compute，Atari
score 不能证明物理 latent；小物体如 Video Pinball 的球会被 reconstruction model 漏掉。对 TDWM，
它是反对“加 RL reward/value 就自然更物理”的实证：reward gradient 贡献小且有时有害。RL
模块默认不更新 world model，这个 frozen-WM 对照应成为我们实验的必选项。

#### DreamerV3（`F`）

**主来源与发表状态。** 预印本题为 [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)；
最终正式版本题为 [Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2)，
Danijar Hafner、Jurgis Pasukonis、Jimmy Ba、Timothy Lillicrap，Nature 640, 647--653，2025 年
4 月发表。旧报告只写“2023 arXiv”或沿用预印本标题会漏掉正式发表状态。

**故事与统一模型。** V3 不靠每个 benchmark 调 loss，而用一组 scale-robust 技术让同一配置跨
Atari、ProcGen、DMLab、DMControl、BSuite 与 Minecraft。RSSM 保持 discrete stochastic state+
GRU memory，predict image/vector input、reward 和 continuation。world-model loss 分 prediction、
dynamics、representation 三块：`L_dyn=max(1 nat, KL[sg(q)||p])` 主要训 prior，
`L_rep=max(1,KL[q||sg(p)])` 主要训 posterior，representation weight 0.1；categorical probabilities
混入 1% uniform 防零概率，vector inputs 先 symlog。

**actor-critic 与梯度。** critic 不回归单一 scalar，而把 symlog value two-hot 到 `[-20,20]` bins，
再 symexp 取期望；lambda-return 用 `gamma=.997`，imagined horizon 约 15。critic 同时在 imagination
和 replay states 上训，后者 scale .3，并用 EMA critic。actor 改为纯 REINFORCE，advantage 除以
`max(1, EMA(p95-p5))`，低 signal 时不人为放大噪声，entropy `3e-4`。LaProp、adaptive gradient
clipping、RMSNorm/SiLU、block GRU 等共同稳训练。actor/critic loss 不更新 world-model parameters；
RL 只通过在线行为改变 replay distribution。

**跨域数字。** 总计 150+ tasks，通常 5 seeds（ProcGen 1 seed，BSuite/Minecraft 10）。Atari
57 at 200M frames：gamer median/mean `830%/3381%`，MuZero `693/3054`；ProcGen 16 at 50M
normalized mean `66.01`，PPG `64.89`、PPO `42.80`；DMLab-30 at 100M human-capped mean
`71.4`，IMPALA 1B steps `66.3`。Atari100k Dreamer mean/median `125/49%`，EfficientMuZero
`190/109`，所以并非所有协议最强。18 个 proprio tasks、500k steps 平均 `871`，D4PG `792`；
20 visual tasks、1M 平均 `861`，DrQ-v2 `770`。Minecraft 100M steps、10 seeds return `9.1`，
IMPALA `7.1`、Rainbow `6.3`、PPO `5.1`；所有 runs 曾找到 diamond，但预算末 diamond 仅约
`.4%` episodes。默认约 200M 参数、单 A100；论文另扫 12--400M，规模与 replay 越大整体越好。

**消融、局限与 TDWM 关系。** Figure 6/Extended Fig. 17 的 14-task ablation 中，各 robustness
组件都有贡献，KL objective 最大，其次 return normalization 与 two-hot；learning-signal ablation
再次显示 unsupervised reconstruction 比 reward/value 更关键。局限是所谓“一套超参数”仍有
benchmark-specific replay ratio、action repeat、env instances 与 budgets；很多 baseline 来自文献
而非统一重跑，ProcGen 只有 1 seed，BSuite DeepSea exploration 仍接近 `.01`。它没有证明
reward-free revaluation，反而再次说明 dense self-supervision 主导。对 TDWM 可借 free-bits、
two-hot、normalization 等稳定技巧，但不能把 DreamerV3 成绩当“RL gradient 防 LeWM collapse”
证据。

#### Dreamer 4（`F`）

**主来源与发表状态。** [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)，
Danijar Hafner、Wilson Yan、Timothy Lillicrap，2025 年 9 月 arXiv v1。当前未核到正式主会或
期刊接收，故 Dreamer 4 仍是预印本；标题也不是 “Dreamer 4” 本身。

**故事与三阶段数据流。** 前三代 Dreamer 的 RSSM 很快但难拟合开放视觉分布；大型视频模型
逼真却太慢，且交互物理不准。Dreamer 4 用 400M causal tokenizer 加 1.6B dynamics transformer：
阶段一在 action-labeled 和 unlabeled video 上预训练 tokenizer/dynamics；阶段二插入 task tokens，
继续 video loss并以 behavior cloning 学 policy、reward；阶段三只在模型 imagination 中做 offline
RL。tokenizer 是 block-causal transformer，每帧 patch 与 latent tokens 编码后经低通道 tanh
bottleneck，decoder 重建；式 (5) 为 MSE+`0.2 LPIPS`，每图 patch dropout `p~U(0,.9)` 做 MAE，
随后 tokenizer 冻结。

**shortcut-forcing world model。** dynamics 输入交错的 action tokens、tokenizer representations、
registers、noise-level 与 step-size tokens；无 action 的视频用 learned missing-action embedding。
普通 diffusion forcing 的 velocity output 会把高频小误差逐帧累积，作者改为预测 clean `z1`
（x-prediction）。最小 step 用 clean-target MSE；较大 step bootstrap 两个 half steps并 stop-gradient，
再乘 `(1-tau)^2` 变回 x-space尺度；`w(tau)=0.9 tau+0.1` 把容量放到高 signal levels。推理每帧
只用 4 sampling steps，过去 context 轻微加噪 `tau_ctx=.1`。axial space/time attention、每 4 层
才做 long temporal attention、GQA、QK norm、register tokens 与长短 batch 交替共同做到实时。

**BC 与 imagination RL 的梯度边界。** task token 能 attend 所有模态，其他模态不能反向 attend
task，避免 dynamics 偷看任务。阶段二式 (9) 对未来 8 个 action/reward 做 multi-token NLL，reward
用 symexp two-hot，并同时继续 dynamics loss。阶段三从 dataset context 每个只起一条 rollout，
模型生成 representations、policy 采动作、reward/value heads标注；lambda-return discount `.997`。
PMPO 只看 advantage 符号，正负样本各自平均，`alpha=.5`，并加 `beta=.3` 的 reverse-KL 到 frozen
behavioral prior，限制 offline model exploitation。关键原文是：默认 **只更新 policy/value heads，
冻结 transformer**。脚注说全 transformer finetune只有小收益且必须同时保留 dynamics/prior/reward
loss。因此不能说 RL 训练了 Dreamer 4 的 world model。

**Minecraft 结果。** VPT contractor subsets 6--10 共 2541 小时、360×640、20 FPS，90/10 split；
键盘为 binary vector，鼠标 mu-law 后每轴 11 bins、组合 121 类。相对 VPT 使用 270k 小时合成
action web video，Dreamer 4 只用这 2.5k 小时。1000 个 60-min 随机空世界 episode 的 Table 7：
Dreamer 4 log/stone-pickaxe/iron-ingot/iron-pickaxe/diamond 成功率为
`99.1/90.1/39.5/29.0/0.7%`；WM+BC 为 `99.6/89.4/27.8/16.9/0`，Gemma-3 VLA 为
`98.5/76.7/22.5/11.2/0`。所以 imagination RL 对铁镐有实增益并首次离线拿 diamond，但 diamond
只有 7/1000 量级；成功 episode 平均约 20.7 分钟。Table 1 单 H100：Dreamer 4 `21 FPS`、9.6s
context、16 项人工交互任务 14 成；Oasis large 约 5 FPS、1.6s、5/16，Lucid-v1 44 FPS 却 0/16。
这项人工评测非盲测、单操作者，不等同自动 benchmark。

**action grounding 与模型消融。** 所有模型看 2541 小时视频，只给 0/10/100/1000/2541 小时
actions；16-step prediction、320-frame context 下，10 小时 action 达到 full-action model 相对
区间的 `53% PSNR / 75% SSIM`，100 小时达 `85/100%`。只在 Overworld 给 action、Nether/End
只给无标签视频，OOD action conditioning 仍为 full-action 的 `76% PSNR / 80% SSIM`。Table 2
的 48-hour cascade：64-step diffusion baseline `0.8 FPS/FVD306`；直接 4 steps `9.1/875`；shortcut
`9.1/329`；x-loss `9.1/151`；ramp `9.1/102`；完整 256-token model `21.4 FPS/FVD57`。
v-space 完整架构 FVD `124`，支持 x-space 对长 rollout 的贡献。

**优点、局限与 TDWM 关系。** 优点是 offline RL、可交互高容量 video WM、action-label efficiency
和 objective/architecture cascade 都做得罕见地完整。限制是 2B 参数、256--1024 TPUv5p 训练、
单一 Minecraft 主域；模型自己承认只有 9.6 秒记忆、inventory 会漂移，离完整游戏 clone 很远。
offline policy 仍可能利用 learned reward/dynamics，behavior prior 只是约束；diamond `.7%` 也不能
被写成可靠掌握。它证明“强预训练 world model 能作为离线 RL simulator”，不是“RL 让 world
model 更物理”。对 TDWM，最值得复用的是 task-to-dynamics 的 causal attention barrier 与 frozen-
WM RL 对照；若目标是让 RL 真正改善 LeWM，必须证明更新 encoder 后 predictive/OOD metrics
也改善，而不只是 policy return 上升。

#### SimPLe（`F`）

**发表状态与故事。** [SimPLe](https://arxiv.org/abs/1903.00374) 正式发表于 ICLR 2020。它要回答的
不是“latent 是否有物理意义”，而是一个更早、更直接的问题：Atari 只允许约十万次真实交互时，
能否先学一个视频模拟器，再把昂贵的真实交互替换成大量模型内 PPO 训练。因而它的主线是
sample efficiency，而不是 reward-free 表征或 zero-shot control。

**模型和训练循环。** 模型读入最近四帧与动作，预测下一张 RGB frame、reward 和 episode
termination。未来并非完全确定：训练时另有一个看到真实未来的 posterior，把不确定因素压成
离散 latent；测试 rollout 时则由只看过去的 autoregressive prior 产生该 latent，并用
straight-through estimator 训练。像素采用 categorical likelihood 和 clipped loss，scheduled
sampling 从 teacher forcing 逐渐过渡到模型自身输出。真实交互、重训模型、在模型中训练 PPO
共循环 15 次；每轮收集 6,400 个真实 environment steps，总计 102,400 steps，也就是 Atari
frame-skip 口径下 409,600 frames。

**RL 到底怎样“帮助”模型。** PPO 的梯度没有反向进入 video model。RL 的作用是改变下一轮
真实数据的访问分布：策略走到旧模型没覆盖的位置，真实 rollout 把这些错误区域重新放进 replay，
模型再做监督学习。这是一种 Dyna 式 data aggregation，而不是“用 value loss 防止 encoder
坍塌”。为限制 model exploitation，每条 imagined rollout 约 50 步便截断，从真实 replay frame
重新开始，并在截断位置用 value bootstrap；总 imagined interaction 约 1,520 万步。

**实验读法。** 论文评测 Atari 100k 的 26 个游戏、5 次运行。后续版本专门补充了与经过调参的
model-free PPO 的公平比较：两者各在 13/26 个游戏领先，而不是一个压倒性胜利。消融显示随机
latent 优于纯确定性模型，rollout horizon 25 到 50 较稳、100 明显受累于误差累积；持续加入新
真实数据也比只学一次模型重要。代价同样很大：模型内训练远多于真实交互，而且像素生成会把大量
容量花在控制无关细节上。

**优点、缺点与本项目关系。** 它的优点是很早就把“模型误差会被策略主动利用”当作系统问题，
并用短 rollout、重启和迭代采数正面处理。缺点是任务 reward、在线交互和像素 reconstruction
全部绑定在一起，结果没有检验 latent rank、跨 reward 迁移、反事实 action accuracy 或物理变量
可读性。对 TDWM 最有用的不是照搬视频生成器，而是把 RL 的两种作用分开：一是只改变采样分布，
二是让 RL 梯度直接进入表示。SimPLe 只支持前者。

#### IRIS（`F`）

**发表状态与故事。** [IRIS](https://arxiv.org/abs/2209.00588) 正式发表于 ICLR 2023。它沿着
SimPLe 的问题继续走，但把“直接回归下一张图”改成“先把图像离散成词，再用 Transformer 像
语言模型一样预测视觉词”。作者希望 categorical token 能比连续像素回归更自然地表达多模态
未来，并让统一的序列模型同时承担视觉 dynamics、reward 和 termination prediction。

**完整数据流。** VQ autoencoder 先把一帧图像压成默认 16 个离散 tokens；训练目标由 L1
reconstruction、codebook commitment 和 perceptual loss 等权组成，离散选择通过
straight-through estimator 更新。随后 GPT 接收历史 frame tokens 和 actions，在一帧内部也按
token 顺序自回归预测下一帧，并额外输出 reward 与 done。actor-critic 使用解码后的 imagined
history，按 DreamerV2 风格从真实 replay 中的状态起步，通常想象 20 步，再用
`gamma=0.995`、lambda-return、entropy regularization 更新策略和价值。

**梯度关系。** world model 仍由 reconstruction/token likelihood、reward 和 termination targets
训练；actor/value 在 imagined trajectories 上优化，不把策略梯度直接写成 tokenizer 或 GPT 的
anti-collapse loss。RL 主要决定收集什么真实数据，以及在已经学到的生成模型里如何行动。因此，
IRIS 展示的是“world model 支持 RL”，并不能直接倒推出“RL 使 world model 更物理”。

**实验和数字。** Atari 100k、26 个游戏、5 seeds 上，论文报告 mean/median human-normalized
score 为 1.046/0.289，IQM 为 0.501，optimality gap 为 0.512，并在 10 个游戏超过人类。SPR
对应为 0.616/0.396、IQM 0.337、gap 0.577、6 个游戏超过人类。这里必须同时看 mean 和 median：
IRIS 的 mean 很高，但 median 低于 SPR，说明少数特别强的游戏会抬高平均值。默认 16 tokens
会漏掉小物体；增到 64 tokens 能明显改善 Alien、Asterix、BankHeist 等任务，却增加序列长度与
计算。论文规模约为 8 张 A100 40GB、3.5 个 environment-days。

**优点、缺点与本项目关系。** 优点是端到端系统完整，多模态视觉未来不再被一个高斯均值强行
抹平，也认真报告了分布式统计。缺点是逐 token 生成慢、误差会在 token 与时间两个轴上累积；
VQ reconstruction 仍可能优先保留纹理而非控制状态，reward/done heads 又使表示任务化。它可以
作为“生成式、离散、在线 world model”的强对照，却不是 LeWM 所追求的轻量 reward-free latent
dynamics，也没有证明 latent 的物理充分性。

#### MuZero（`F`）

**发表状态与故事。** [MuZero](https://www.nature.com/articles/s41586-020-03051-4) 正式发表于
Nature 2020。它刻意挑战“规划前必须学会重建环境”这一传统：若模型只服务于搜索，那么只要能
预测搜索真正使用的 reward、policy 和 value 就够了，内部 latent 不必对应棋盘像素或 Atari 的
可解释物体。这个故事非常成功，但也明确把“决策有用”和“物理真实”分开了。

**三个函数怎样配合。** representation `h` 把 observation history 编成初始 latent `s_0`；
recurrent dynamics `g(s_k,a_k)` 产生下一 latent 与一步 reward；prediction `f(s_k)` 输出 policy
logits 和 value。MCTS 在 latent tree 中反复调用 `g/f`，把 visit counts 变成改进 policy，再执行
根节点动作。与 reconstruction model 不同，搜索期间不需要把 latent 解码成 observation。

**训练和梯度。** replay 中取一段真实 action sequence，从 `h` 得到根状态后用 `g` 展开默认
5 步；每个展开位置同时拟合真实 reward、n-step bootstrapped value target 和当时 MCTS 的 visit
distribution。三个网络在这三个损失下联合反向传播，并加参数正则。也就是说，模型被训练成
value/search-equivalent，而不是 transition-likelihood-optimal。目标依赖当前搜索与策略，数据和
targets 又随 agent 进步不断变化。

**实验应该怎样理解。** 围棋、国际象棋和将棋使用每步约 800 次 search simulations；Atari 使用
约 50 次。完整 Atari 结果建立在 200 亿 frames 这一巨大预算上，论文表中的 median/mean
human-normalized 百分数约为 2041.1/4999.2。MuZero Reanalyze 把交互降到 2 亿 frames 后约为
731.1/2168.9，但仍远不是 Atari 100k。性能随 search simulations 增加到约 100 次后趋于饱和。
这些结果证明了隐式模型加搜索的规模能力，不是少数据或轻量 world model 证据。

**优点、缺点与本项目关系。** 最大优点是把“模型应该为用途服务”落实成强算法，并绕过困难的
像素生成。最大限制也由此而来：latent 可以丢掉对当前 reward/search 无用、却对换任务重要的
物理因素；deterministic recurrent state 对真正随机、多模态未来也没有显式校准。若 TDWM 把 RL
targets 加进 LeWM，MuZero 是必须面对的概念前作；但我们的科学问题应是如何获得 decision
guidance 而不把 reward-agnostic primitive dynamics 缩成单任务的 value-equivalent 模型。

#### TD-MPC（`F`）

**发表状态与故事。** [TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html) 正式发表于
ICML 2022，PMLR 162；旧稿把它写成 ICLR 2022 是错误的。它的核心判断是：短期局部 dynamics
通常较容易学，长时 rollout 容易漂移；因此 MPC 只让 world model 负责短 horizon，再用 model-free
Q 在末端概括剩余回报。论文把这套组合称为 task-oriented latent dynamics（TOLD）。

**模型和目标。** observation encoder `h`、deterministic latent dynamics `d`、reward model `R`、
双 Q 与 policy prior `pi` 共享一段 replay unroll。默认多步损失的时间权重为 `lambda=0.5`，包含
reward MSE、TD Q MSE 和对 EMA target encoder 的 latent consistency，系数分别约为 0.5、0.1、
2.0；预测 latent 承接这些梯度。actor 在 stop-gradient latent 上最大化 Q，因此直接更新 policy，
但不通过 actor loss改写 encoder。这里显式 anti-collapse 依赖 EMA target 和多任务监督，而不是
RL loss 的自然保证。

**规划过程。** planner 从 Gaussian action sequences 采样默认 512 条，并混入约 5% policy
proposals；每轮保留 64 个 elites，以 exponentiated value weighting 更新分布，共做约 6 轮。目标
是 horizon 内 predicted rewards 加 terminal Q。horizon 从 1 逐渐 anneal 到 5，并用上一步计划
warm start；每个真实时刻只执行第一个 action，再重新规划。默认报告约 20ms/step，说明短 horizon
是精度与实时性的折中，而非声称模型可无限 rollout。

**实验和失败处。** 论文在 DeepMind Control、pixel control、Meta-World 等共 92 个任务上比较，
总体强调同一框架兼顾 state/pixel 与多种 action spaces。它在大多数常规任务表现强，但 Finger
Turn Hard 等 hard-exploration 任务可落后 SAC/LOOP，说明高质量局部模型和 planner 不能自动
解决覆盖不足。消融支持 reward、Q、consistency 和 policy proposal 的组合，而不是某一个 loss
独自解释成绩。

**优点、缺点与本项目关系。** TD-MPC 很清楚地展示了“primitive local model + long-tail value”
的互补性，是我们想法的直接结构前作。它的限制是 reward/Q 完全绑定当前任务，deterministic
latent 不表达随机未来，也没有跨 reward 的冻结 world-model 评测。TDWM 若提出 RL 帮助 LeWM，
必须比较它到底只是复现 TD-MPC 的 terminal value、policy proposal 或 task-oriented representation，
还是能在这些控制收益之外保留 reward-free、可迁移的物理动态。

### 14.12 决策感知模型学习：RL 怎样直接改变模型目标

#### Value-Aware Model Learning / VAML（`F`）

**发表状态与故事。** [VAML](https://proceedings.mlr.press/v54/farahmand17a.html) 正式发表于
AISTATS 2017，PMLR 54。maximum likelihood 平均拟合所有 transition error，但 planner 实际只
通过 value function 查询模型。作者因此问：容量有限时，能否把模型的误差预算集中到真正影响
Bellman backup 的方向，而不必恢复完整 dynamics？

**目标。** 给定 planner 可能产生的 value-function class `F`，VAML 让真实 transition 与模型
transition 对这些 functions 的期望尽量一致。典型 robust 形式是：

```math
\mathbb E_{s,a}\sup_{V\in\mathcal F}
\left|\mathbb E_{s'\sim P}[V(s')]-\mathbb E_{s'\sim\hat P}[V(s')]\right|^2.
```

`F` 取 1-Lipschitz functions 时与 Wasserstein discrepancy 相连；在线性 value 与 exponential-
family model 等设定下，论文给出可计算梯度和有限样本界。它比较的是 expectation through model，
并非要求模型 sample 与真实 next state 一一对齐，因此允许一个在状态预测上“不真实”、但
Bellman backup 正确的模型。

**实验究竟有多大。** 实证只有一个 25-state random-walk MDP，不是视觉控制。作者人为限制
transition model 和 value approximation 的 partition capacity，用约 5,000 个 sampled transitions
训练，并在 approximate value iteration 中比较 VAML 与 MLE。模型 misspecified 时，VAML 往往
得到更小的 value error；但当模型 partitions 恰好对齐环境结构时，MLE 可以追平甚至更好。有限
样本噪声也会缩小理论优势。这个实验主要是机制示范，不能直接外推到深度 world model。

**优点、缺点与本项目关系。** 优点是首次把“模型误差是否重要”写成 planner-relative objective，
而不是事后用 return 猜测。限制是 minimax 优化可非凸、function class 很难选，并且一旦换 reward
或 planner，原本被忽略的 dynamics 方向可能重新重要。它是“RL/value 直接改变 model loss”的
基础前作，也提醒我们：控制更好不等于物理更真；若要保留 zero-shot revaluation，value class 必须
覆盖足够多任务，并与纯 self-supervised loss 做受控比较。

#### Iterative VAML（`F`）

**发表状态与故事。** [IterVAML](https://papers.nips.cc/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html)
正式发表于 NeurIPS 2018。原始 VAML 对整个 function class 做最坏情况匹配，计算重且可能过度
保守；IterVAML 改为只跟随 approximate value iteration 实际走过的 value sequence，使模型容量
围绕当前规划过程逐轮分配。

**算法怎么走。** 第 `k` 轮先用当前 policy 收集 transitions；拟合模型，使真实 next state 上的
`V_k(s')` 与模型 next-state distribution 下的 value expectation 接近；再在新模型上做一次
approximate value-iteration update 得到 `V_{k+1}` 和新 policy。随后重新采数、重新拟合。模型
目标因此是一个随 value/policy 改变的 moving target，而不是固定的 one-step likelihood。

**理论和必须纠正的事实。** 论文给出 finite-sample model-learning error 以及该误差沿迭代传播到
value 的分析，但这些结论依赖所选 function/model classes、采样分布和每轮近似误差。更重要的是：
论文没有任何 empirical experiment。结论明确把实证研究列为 future work。旧稿写“实验尚未完全
复核”仍会让人误以为正文存在实验，正确说法是“这篇理论论文没有实验”。

**优点、缺点与本项目关系。** 优点是比一次性 VAML 更贴近实际 planner 的函数轨迹，也把
policy-induced distribution shift 纳入循环。缺点是模型强烈依赖当前 reward、value path 和采样
coverage；换 reward 或换 planner 时，过去没被查询的物理方向可能完全没有学。它为 RL 反复
指导 model 提供了理论先例，却不能作为视觉 LeWM 有效性的实验证据；TDWM 若采用类似交替训练，
必须补上稳定性、跨任务遗忘与模型真实性诊断。

#### Policy-Aware Model Learning / PAML（`F`）

**发表状态与故事。** [PAML](https://arxiv.org/abs/2003.00030) 是 arXiv 预印本；截至本次核查
未找到正式主会或期刊版本。它比 VAML 更进一步：如果模型最终只用来更新 policy，那么模型
不一定要保 value expectation，而可以直接保住“下一步 policy gradient 往哪里走”。

**目标与循环。** 其理想目标是让真实环境和模型环境产生的 policy gradients 接近，例如

```math
\|\nabla_\theta J_P(\pi_\theta)-\nabla_\theta J_{\hat P}(\pi_\theta)\|_2^2.
```

论文为 REINFORCE 与 deterministic policy gradient 推导样本近似，并放入 Dyna 循环：当前 policy
收集真实数据，PAML loss 拟合模型，policy 再在模型里更新。目标同时依赖当前 visitation、reward、
value/critic estimate 和 policy 参数；policy 漂移后，旧模型的 PAML loss 可重新升高，因此每轮
model update 与 policy update 次数成为敏感超参。

**实验到底支持什么。** 小型 2/3-state tabular MDP 在严格 model-norm/capacity constraint 下，
PAML 比 MLE 更能保住正确 gradient；放宽容量后两者趋同。连续实验包括 LQR 的 REINFORCE，
以及 Pendulum、Swimmer、Hopper、HalfCheetah、Ant 上的 DDPG。LQR 加入 irrelevant dimensions
时 PAML 更稳，但无 distractor 时 MLE 可更快。作者自己明确承认：tabular 的明显优势没有稳定
迁移到 MuJoCo，PAML 与 MLE 总体相近；短 `H=1` model rollout 也可能掩盖差别。因而不能把这篇
写成“已证明 policy-aware loss 在复杂控制显著胜过 MLE”。

**优点、缺点与本项目关系。** 它是“RL gradient 教模型哪些误差重要”最直接的前作。问题是只
保证当前 policy update direction，不保证 action-sequence rollout、goal replacement、policy 外
query 或物理因子保留；gradient estimator 本身的噪声也会污染 model target。若 TDWM 让 actor
gradient进入 LeWM，必须将 PAML 作为概念与实验 baseline，并重点测换 reward、换 policy 和 frozen
encoder 后是否发生任务过拟合。

#### Value Gradient weighted Model Learning / VaGraM（`F`）

**发表状态与故事。** [VaGraM](https://arxiv.org/abs/2204.01464) 正式发表于 ICLR 2022。它针对
VAML 的实际失败：若直接在模型自己预测的 OOD next state 上查询 learned value，value surface
可能产生许多错误的 iso-value minima，模型便会学会“钻 critic 的空子”，甚至不再接近真实状态。

**推导与实现。** 对真实下一 state 邻域的一阶 Taylor expansion 表明，transition error 沿
`∇V(s')` 大的方向更影响 value。为避开 VAML 在模型 OOD state 上查询不可靠 value 和
iso-value spurious minima，VaGraM 使用稳定的 diagonal upper-bound 形式：

```math
\mathcal L_{VaGraM}\approx
\|\operatorname{diag}(\nabla_{s'}V(s'))(\hat s'-s')\|_2^2.
```

它仍以真实 `s'` 为 supervised center，只把各坐标误差按 value sensitivity 加权，因此不像原始
VAML 那样允许模型跑到任意等值面。这个 diagonal Cauchy upper bound 易于放进 Dyna-style
actor-critic，却不是原始 value error 的完全等价式。

**实际数据流与控制。** replay transition 先更新 critic；在真实 `s'` 处对 critic 求 state gradient，
对该权重 stop-gradient 后更新 transition model，避免 model 为降低 loss 反过来操纵 critic。随后从
真实 replay states 起短 model rollouts，actor/critic 按原 Dyna/MBPO 流程学习。测试时执行 actor，
VaGraM 本身不增加新 planner；因此它改变的是训练 model 时各 state coordinate 的权重，而不是
额外提供一个推理期 value head。

**实验怎样读。** Pendulum 的可视化显示 IterVAML 容易向 OOD minima 发散，VaGraM 仍靠近真实
transition。Hopper 容量/干扰维实验中，大网络且无严重 distractor 时各方法差别不大；网络变小或
加入 15 个无关维度后 VaGraM 明显更稳，20 个维度时两者都可能失败。更广的 MuJoCo 对比总体
多为相当而非全面碾压。论文多处使用约 16 runs，真正证据是“容量受限和 distractor 下更稳”，
不是所有环境都提升。

**优点、缺点与本项目关系。** 它保留 supervised anchor，工程上比 minimax VAML 稳，也诚实
揭示了 value model exploitation。限制是一阶局部假设、critic gradient 质量和坐标系依赖：latent
做任意旋转后 diagonal weighting 会改变，`grad V` 为零的维度也可能被完全忽略。若给 LeWM 加
value-gradient loss，VaGraM 是最直接 baseline；同时必须测它是否只让 latent 坐标迎合当前 value，
而没有提高 reward-free dynamics 或跨任务物理表示。

#### Transition Occupancy Matching / TOM（`F`）

**发表状态与故事。** [TOM](https://proceedings.mlr.press/v211/ma23a.html) 正式发表于 L4DC 2023，
PMLR 211；旧稿把 venue 写成 UAI 2023 是错误的。它认为 uniform one-step model fitting 浪费容量：
真正影响当前 policy return 的，是该 policy 实际访问到的 `(s,a,s')` transition occupancy。

**理论对象。** 定义当前 policy 下真实/模型 transition occupancy：
`d_T^pi((s,a),s')=d_T^pi(s,a)T(s'|s,a)`，并最小化两者的 f-divergence，而不是均匀拟合 replay
中的所有 transition。论文用 Jensen inequality 将真实 return 写出一个包含 model reward 与
occupancy f-divergence 的 lower bound，从而解释为何匹配该分布有利于 policy optimization。

**实际算法。** 直接 rollout 长模型来估 occupancy 会累积误差，所以实现借用 SMODICE 的 dual
form：分类器估计当前-policy transitions 与 replay 的 density ratio，再用 relevance Q/dual variable
把比率转成 sample weights，最后做 weighted maximum-likelihood transition fitting。policy 在模型
中更新，再与真实环境交互形成循环。它不是训练一个 successor generator，而是用 occupancy 信息
重新加权 local model。

**实验和证据。** 论文先用 road-and-rocks 说明容量应放到访问区域，再在 5 个 MuJoCo 环境、
300k real steps、4 seeds 上接入 MBPO。aggregate 上 TOM 在 4/5 个任务最好；Humanoid 的提升可
超过 MBPO 约 60%。把权重简单换成 recency 并不能得到同样结果，支持“policy relevance”而非
“多看新数据”的解释。不过环境仍是低维 state、在线单任务控制，没有视觉表征实验。

**优点、缺点与本项目关系。** TOM 清楚地区分了数据里“有 transition”和当前策略“会用到该
transition”。限制是 current-policy support 外的 action/state 可被有意忽略，policy 大幅改变或换任务
后模型可能失效。它与 TD-Flow/Jumpy 的 occupancy model 不是一回事：TOM 用 occupancy 加权
primitive transition，后者直接预测长期 occupancy。TDWM 若用 RL 来选择 LeWM 训练样本，应把
TOM 当作 sampling/weighting baseline，而不是把收益误归因于新 latent objective。

#### Value Equivalence（`F`）

**发表状态与故事。** [The Value Equivalence Principle](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
正式发表于 NeurIPS 2020。它不是一套固定神经网络，而是回答“什么时候一个错误的 dynamics
model 对规划仍然足够”。作者把模型质量从 state prediction 改成 Bellman-operator behavior，给
MuZero、Predictron、Value Prediction Networks 等隐式模型一个统一解释。

**定义和核心结论。** 给定 policy 集 `Π` 与 value-function 集 `V`，若对所有
`π ∈ Π, v ∈ V` 都有
`T^πv = T_hat^πv`，两个模型在该集合上 value equivalent。扩大 policy/value 集会收紧可接受
模型类；使用所有 policies/functions 时，要么 pin down true model，要么 hypothesis class 中没有
可行模型。对有限集合，只需覆盖 `Π` 的 pointwise span 与 `V` 的 linear span；有限 MDP 中适当
basis 即可代表更大集合。论文还给出 value-equivalent model space dimension 随独立 policies 与
functions 乘法收缩的结果。

**怎么训练。** 经验 loss 不再最大化 next-state likelihood，而是从真实 transition samples 估计
`T^πv`，再让模型 Bellman update 与之接近。选择哪些 `π` 和 `v`，等价于明确告诉模型未来会被
怎样使用。这个设计可能在容量受限时舍弃无关 transition detail，但也可能因为选择太窄而失去
replanning 与换 reward 能力。

**实验范围。** Four Rooms、Catch 和 Cart-pole 先用随机 policy 收数据，再比较 MLE 与 VE 模型
用于真实环境控制。前两者使用 rank-constrained transition matrices；Cart-pole 的 transition/value
都用 neural networks，并限制层间 rank/hidden capacity。固定模型容量或固定 function-set size 的
多组结果里，VE 尤其在容量紧时优于 MLE；误差条是 30 runs 的一个标准差。实验是小域原理验证，
没有像素、复杂随机 dynamics 或 broad benchmark。

**优点、缺点与本项目关系。** 优点是把“useful model”严格定义成相对于 planner 的等价关系，
避免含糊地说 latent 更好。缺点是 value equivalence 绝不等于 physical equivalence、causal
identifiability 或 task-general world model。TDWM 若只用 return 证明 RL-assisted LeWM 更好，
最多说明某个 planner/task 下更 decision-useful；若要声称物理意义，仍需 action-conditioned
rollout、换 reward、OOD factors、controllability 与 probe 等独立证据。

#### Calibrated Value-Aware Model Learning（`F`）

**发表状态与故事。** [Calibrated VAML](https://proceedings.mlr.press/v267/voelcker25a.html) 正式
发表于 ICML 2025，PMLR 267。论文发现一个容易忽略的问题：stochastic environment 中，常见
sample-based VAML/MuZero surrogate 含有 target variance；即使无限数据并达到 population optimum，
也可能偏好低方差的错误 model/value。这不是优化没做好，而是 loss 本身未校准。

**统一视角和修正。** 作者用 `(m,b)` 表示模型展开 `m` 步、真实 Bellman bootstrap `b` 步：
IterVAML 类似 `(1,0)`，MuZero/TD-MPC 风格常有 `b>=1`，model-free 是 `m=0`。他们证明 sample
loss 与理想 expectation-matching objective 相差一个 model-sample variance 项；校准版显式减掉
这个偏差，通常至少需要两个独立 model samples。一个 deterministic one-step model 有时仍可对
某个 value equivalent，但不能因此宣称它恢复了真实 stochastic transition distribution。

**估计与梯度。** 对同一 `(s,a)` 从 learned stochastic model 独立采至少两个 next states，以
样本间差估计 model-side variance，再从原 surrogate 中减去该项；校正后的 Monte Carlo estimator
才与理想 expectation loss 对齐。梯度更新 value-aware model及其共享 latent path，target/value
分支按具体 `(m,b)` 变体使用 stop-gradient。论文提供的是 objective correction，不规定一个新的
test-time solver；控制仍沿用被校准算法原有的 TD-MPC/actor/planner。

**实验和反直觉结果。** tabular Garnet MDP 使用 50 states、每个 state-action 10 个 successor，
系统改变 stochastic temperature 与 model rank，验证未校准目标的偏差。深度实验构造 TD-MPC
风格 deterministic/Gaussian latent models，在 hard DMC dog 和 humanoid tasks 上运行约 30 seeds、
2M steps。校准对 `(1,1)` 尤其重要；dog 中 stochastic model 常有帮助，humanoid 却可能是
deterministic `(1,0)` 最好。去掉 BYOL/next-latent auxiliary 会让 humanoid 完全失败，说明 value
target 仍不能独自塑造稳定 latent。校准通常提高 learned variance/entropy，但不保证每个任务 return
都同步上升。

**优点、缺点与本项目关系。** 优点是把“理论 value-aware”与“有限 sample loss”之间的偏差
拆得很清楚，并用大种子数展示结论并非单次偶然。限制是校准需要多模型采样、增加成本，且模型
distribution 校准仍是相对于指定 values/rewards。对 TDWM 这是直接警告：给 LeWM 加 TD/value
loss 时，必须区分 point prediction、value equivalence 和 stochastic calibration；还应保留独立的
self-supervised dynamics/anti-collapse objective，不能期待 RL target 单独承担一切。

### 14.13 结构化表示与行为表示

#### DeepMDP（`F`）

**发表状态与故事。** [DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html) 正式发表于
ICML 2019，PMLR 97。它要把高维 observation 压成一个仍可当 MDP 使用的 latent：在 latent 中
既能预测 reward，也能预测下一 latent distribution，便可用这两种误差约束 representation，而
不必重建所有像素。

**目标和理论。** encoder `phi`、latent reward model 与 latent transition model 联合训练。理论以
reward error 和真实/latent transition 的 Wasserstein discrepancy 上界 value-function distortion；
无限 horizon 结果需要 Lipschitz dynamics 且 `gamma K_P<1` 之类的 contraction 条件，finite-horizon
版本较宽松。若两种误差为零，某些 value-relevant 非双模拟状态不能随意坍塌。注意理论讨论的是
distribution metric，实际 deterministic Atari implementation 常把它简化成 next-latent L2。

**实验。** DonutWorld 将 32x32 observation 编到 2D，DeepMDP 能恢复环形状态结构并按行为
等价关系合并轨道，普通 autoencoder 更关注像素。Atari 部分覆盖 60 个游戏、3 seeds，把 reward
和 latent-transition prediction 作为 C51 encoder 的 auxiliary losses；部署时仍是 C51，不在 learned
model 中规划。next-latent auxiliary 总体改善 C51，而直接预测 Q logits 的替代可能伤害性能。
所以 Atari 结果证明的是 representation regularization，不是 model-based control。

**优点、缺点与本项目关系。** 优点是很早就把 latent MDP 的 reward/dynamics errors 与 value
差异联系起来，并明确区分 reconstruction 与 control sufficiency。限制是 reward-aware、理论条件
强，实验的 deterministic L2 与 Wasserstein theory 有落差；也没有证明 arbitrary reward 下保留全部
物理状态。它支持“RL task 与 dynamics auxiliary 可共同训练 encoder”，但不支持“RL 自然防
坍塌”。TDWM 应把 reward-free LeWM 与 DeepMDP 式 reward head 分开消融，并做跨 reward 冻结
评测。

#### Deep Bisimulation for Control / DBC（`F`）

**发表状态与故事。** [DBC](https://arxiv.org/abs/2006.10742) 正式发表于 ICLR 2021。它不要求
预测每个像素，而是直接训练 latent geometry：若两个 states 的即时 reward 相同、在相同行动下的
未来行为也相同，它们应该靠近；反之则应该远离。这里的“坍塌”不是所有合并都坏，合并
behaviorally equivalent states 正是目标。

**目标和梯度。** 随机取 replay batch 中两条 transitions，再将 batch permutation 配对，使 latent
distance 逼近 bisimulation metric target：即时
reward difference 加 discount 后的 latent transition Wasserstein distance。Gaussian transition
下用闭式 `W_2`，target encoder/next latent 通过 stop-gradient 稳定训练，再与 SAC 联训。

```math
\|z_i-z_j\|_1 \approx |r_i-r_j|+
\gamma W_2(\hat P(\cdot|z_i,a_i),\hat P(\cdot|z_j,a_j)).
```

transition model 另行预测 stop-gradient next latent；SAC critic loss 也更新 encoder，而 actor 不
更新 encoder。因此稳定性来自 bisimulation target、stop-gradient、dynamics 和 critic 的组合，
不是一个单纯的 RL 防坍塌信号。

**实验与具体数字。** 论文评测 9 个 DMC tasks，含默认背景、简单 distractor 与自然视频背景，
通常 10 seeds。SLAC 在某些默认任务更强，但 DBC 在 distractor 下更稳。把 Walker Walk 的
encoder 迁移到 Stand/Run 只在新 reward 所需 causal ancestors 是原任务子集时合理，作者据此讨论
task transfer 的边界。CARLA 84x420 输入、3 seeds、100k steps 上，成功率 DBC 约 24%，DeepMDP
17%，SAC 12%；平均行驶距离约 179、106.7、123.2，但 crash 指标并非 DBC 全胜。单次运行约需
GTX 1080 12 小时。

**优点、缺点与本项目关系。** DBC 的优点是把 reward、transition 与 latent distance 串成可解释
geometry，并在视觉干扰下显示鲁棒性。缺点是 task-aware bisimulation 会主动丢掉当前 reward 不
需要的物理因素，Gaussian dynamics 与 frame stack 也限制 partial-observation/multimodality。它是
“RL 帮助 representation”最直接的 baseline 之一，却恰好说明：减少表示冗余与保留通用物理状态
可能冲突。TDWM 必须在当前-task return 之外测换 reward 后被合并因素是否还能恢复。

#### MICo（`F`）

**发表状态与故事。** [MICo](https://papers.nips.cc/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html)
正式发表于 NeurIPS 2021。DBC 的 Wasserstein coupling 计算和实现较复杂；MICo 改为两个独立
next-state samples 的 current-policy behavioral distance，让普通 replay TD update 就能训练。

**对象与更新。** MICo 定义当前 policy 下两个
states 的 diffuse behavioral distance：

```math
U^\pi(x,y)=|r^\pi(x)-r^\pi(y)|+
\gamma\mathbb E_{x'\sim P_x^\pi,y'\sim P_y^\pi}U^\pi(x',y').
```

从 replay 独立采两个 next states 即可做 TD regression，target network 提供 bootstrap。参数化
距离用两个 embedding norm 的平均加 `beta` 倍 angle distance，默认 `beta=0.1`；总 loss 是 TD
control loss 与 MICo loss 的加权和，Huber loss 对稳定性重要。作者也定义 reduced angular distance
去掉 self-distance，但论文给出反例：修正后不再保证上界 value difference。

**实验。** Atari 覆盖全部 60 个游戏、5 seeds，MICo 加到 DQN、Rainbow、QR-DQN、IQN、MIQN
上多数提高。12 个 pixel DMC tasks、5 seeds 中，SAC+MICo 总体最强；把 DBC 的 metric 换成
MICo 也能恢复被 DBC 损失伤害的 SAC 表现。证据说明这个 auxiliary 比某些 bisimulation 实现
更稳，但没有模型 rollout、MPC 或 reward replacement 实验。

**优点、缺点与本项目关系。** 优点是算法简单、可扩展到大批任务，并有清楚的 contraction 与
value-bound 理论。限制是 independent coupling 产生 diffuse metric，甚至 `U(x,x)>0`；它只刻画
当前 policy/reward 的 behavioral similarity，不是 action-conditioned simulator。若 TDWM 想用
RL 定义 latent geometry，MICo 是低成本强 baseline；但 success 增益仍不能说明 LeWM 的 primitive
dynamics 或物理意义改善。

#### Contrastive Learning of Structured World Models / C-SWM（`F`）

**发表状态与故事。** [C-SWM](https://openreview.net/forum?id=H1gax6VtDB) 正式发表于 ICLR 2020。
它是纯 reward-free world-model representation 工作：作者认为逐像素生成太浪费，真正有用的是
把场景拆成对象，并学每个对象在 action 和相互作用下怎样移动。其主要问题是 structured latent
dynamics，不是 policy learning。

**结构与损失。** CNN object extractor 把 frame 分成固定数量的 object slots；共享 object encoder
得到 states；
graph neural network 按 action 和其他对象消息预测每个 object 的 state delta。positive energy 是
预测 next slots 与真实 next slots 的平方误差，随机 negative state 通过 margin 被推远：

```math
H=\frac1K\sum_k d\!\left(z_t^k+T^k(z_t,a_t),z_{t+1}^k\right),\qquad
\widetilde H=\frac1K\sum_k d\!\left(\widetilde z_t^k,z_{t+1}^k\right),
\qquad \mathcal L=H+\max(0,\gamma-\widetilde H).
```

负样本来自随机 observation，margin 默认约 1；全连接 object graph 复杂度随 slot 数平方增长。
模型看到两帧以近似速度，action 也按对象因子化。训练后只递归 rollout latent，不训练 reward、
value、actor 或 planner。

**实验和指标。** 数据由 random policy 生成：grid environments 约 1,000 条 x 100 steps，Atari
约 1,000 条 x 10 steps，3-body 约 5,000 条 x 10 steps。评测不是 control return，而是把 1/5/10
步预测 latent 在候选 buffer 中做 retrieval，报告 Hits@1 与 MRR。grid 上接近完美；Atari 方差大且
最优 slot 数依游戏而变，例如 Pong 反而 `K=1` 较好、Space Invaders 需要更多 slots。3-body 的
10-step Hits@1 约 75.5%，autoencoder 约 67.9%。

**优点、缺点与本项目关系。** 优点是 reward-free、对象归纳偏置清楚，并用 multi-step retrieval
而非漂亮 reconstruction 判断 dynamics。缺点是固定 `K`、对象/action factorization、短数据与
简单域假设很强；没有 memory 处理 partial observability，也没有 MPC evidence。它说明“物理
意义”可以来自 object structure，而非 RL；若 TDWM 声称 RL 产生物理 latent，应至少比较一个
不使用 reward 的结构化 baseline，并区分 retrieval 好与 goal-cost calibrated。

#### Structured World Belief（`F`）

**发表状态与故事。** [Structured World Belief](https://proceedings.mlr.press/v139/singh21a.html) 正式
发表于 ICML 2021，PMLR 139。它指出单个 deterministic latent 不足以表示遮挡世界：物体暂时
不可见不等于不存在，而且同一观察可能对应多个真实场景。world model 因此应维护 belief
distribution，而非一条“最佳猜测”状态。

**belief 的内部结构。** 模型维护 `K` 个加权 particles，每个 particle 含 `N` 个 object files；file
分别记录 identity、visibility 和 recurrent state，把“存在”与“当前看见”拆开。图像 encoder 给出
slots，每个 file 通过 interaction GNN 与所有 slots（含 null slot）做概率 matching；可见对象用
glimpse/posterior 更新，不可见对象沿 dynamics prior “imagine”。Sequential Monte Carlo 根据
observation likelihood 更新 particle weights 并 soft-resample，训练最大化 AESMC 的序列证据下界；
连续变量重参数化，离散匹配部分使用 REINFORCE estimator。

**控制和实验。** world belief 先用 random-policy trajectories 训练，再冻结或提供给 A2C/planner。
三个 synthetic partially-observed environments 的预训练数据约为 200k、200k、500k steps，物体
有约 25%-40% 时间不可见；从 `K=1` 增到多 particles 通常改善控制。固定总 simulated-future
预算的 Monte Carlo planning 中，2D Maze return 约为 random -5.69、AESMC `K=30` -2.37、SWB
`K=30` -1.32；3D Food 分别约 -3.39、5.10、12.13。这里的提升针对 object permanence 与 belief
multimodality，不是一般视觉 benchmark。

**优点、缺点与本项目关系。** 优点是把 partial observability、对象身份、不可见 dynamics 与模型
uncertainty 放进统一可训练系统，真正讨论“状态是否充分”。缺点是复杂度随 particles 和 objects
增长，依赖 slot discovery、likelihood/reconstruction 与 synthetic scene assumptions；没有现代
大规模视觉或 LeWM 式简洁 MPC。它提醒 TDWM：如果观测本身非 Markov，再强的 RL loss 也无法
让单帧 latent magically 成为物理 state。history/belief baseline 应与 anti-collapse 问题分开处理。

### 14.14 全文审读后的创新边界

经过 64 篇核心论文的全文审读，下面区分两类判断：“已覆盖”表示已有工作实现了同一接口并给出
直接证据；“相邻但未解决”表示已有组件，尚未证明在保留 LeWM reward-free dynamics 的同时
有效。不能把后一类误写成已经被否定。

1. “用 value/goal geometry 直接重塑 JEPA 的 planner cost”：Value-Guided JEPA 已覆盖；但它的
   joint prediction 版本反而退步，SPR 的 predictor 不用于规划，MuDreamer 也不是 JEPA planner，
   所以“RL 在保住 LeWM dynamics 时还能改善 planner”仍是未解决、必须实证的问题；
2. “多步或开放环训练减少 LeWM rollout drift”：PLDM、RC-aux、Fast-LeWM 已覆盖；
3. “学 temporal distance/reachability 改进 LeWM cost”：RC-aux、Temporal-Distance-JEPA 已覆盖；
4. “跨 horizon consistency 形成长期模型”：Fast-LeWM、UHM、Jumpy 已覆盖不同接口；
5. “普通 SIGReg 不够，修正其 latent geometry”：TC-LeWM、PhyLatent、Metric Non-Collapse 已覆盖；
6. “物理 state/action grounding 让 LeWM 有物理意义”：PhyLatent、PSG-JEPA 已直接覆盖；
7. “测试时用新 transition 自监督适配”：AdaJEPA 已覆盖；
8. “加高层 latent action/subgoal 改善长时规划”：Hi-LeWM、LaWAM 已覆盖，并暴露 support gap；
9. “用 TD 学 reward-free 长期 future 再 zero-shot control”：`gamma`-model、TD-Flow、TD-JEPA、
   RLDP 和 Jumpy 已形成完整谱系；
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

1. “我们首次把 RL/decision loss 与显式 anti-collapse 组合起来。”SPR、MuDreamer、R2-Dreamer、
   VAML 与 TD-MPC 系列已经给出大量相邻组合；但它们并没有证明 **RL 单独**足以防坍塌：SPR
   没有正交拆开 RL 与 augmentation/target，MuDreamer 依赖 BatchNorm，R2-Dreamer 依赖 Barlow
   式 redundancy reduction，TD-MPC 依赖 EMA consistency 与多头监督。真正未决的是 RL 提供的
   独立增量，以及该增量会不会牺牲 reward-agnostic dynamics。
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

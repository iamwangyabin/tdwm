# 用无奖励 Bellman 目标校准 LeWM 的研究方案（草案）

创建日期：2026-08-12<br>
当前状态：研究假设与实验设计阶段，尚未实现或验证<br>
暂定方法名：**Bellman-Calibrated LeWM（BC-LeWM）**

## 1. 结论摘要

“使用强化学习帮助 LeWM”是有意义的研究方向，但不能只表述成“加入 TD loss 防止
特征坍塌”。LeWM 已经使用 SIGReg 处理表示坍塌，并报告了潜在表示中的物理状态结构；
TD-MPC2、TD-JEPA 和近期 value-guided JEPA 工作也已经证明，价值或时序差分信号可以
参与潜在表示学习。因此，简单的 `LeWM + critic` 很难形成清晰的新贡献。

本项目更值得检验的研究问题是：

> 在完全离线、训练时无外部奖励的条件下，能否从一族伪任务和策略中学习 Bellman /
> successor functionals，并用这些 functionals 校准 LeWM 的编码器和可滚动动力学，使
> 模型在保留通用预测能力和物理结构的同时，对长期控制和未见任务更有用？

推荐的核心方法不是让 LeWM 直接优化某个任务的回报，而是要求模型预测的下一潜在状态
与真实下一潜在状态，在一组 reward-free Bellman functionals 下近似等价。这把 RL
信号变成对 world model 的控制相关误差度量，而不是附加在模型旁边的独立策略模块。

该方向有论文潜力，但文献空间已经拥挤。只有同时满足以下条件，才足以支持主要主张：

1. TD/Bellman 梯度确实改善了 LeWM 的编码器或动力学，而不只是训练出更强的 planner；
2. 改进能迁移到训练期间未见的目标或 reward，而不局限于单一任务；
3. 在相同数据、规划器和计算预算下改善长期规划，并且不破坏一步预测与物理 probe；
4. 相比 value-guided JEPA、reachability auxiliary 和 temporal-distance 方法具有明确差异。

## 2. 问题定义

### 2.1 LeWM 已经解决了什么

LeWM 从无奖励离线轨迹 `(o_t, a_t, o_{t+1})` 中联合训练视觉编码器和一步潜在动力学：

```math
z_t = E_\theta(o_t), \qquad
\hat z_{t+1} = F_\phi(z_t, a_t).
```

其主要目标可以概括为：

```math
\mathcal L_{\text{LeWM}}
= \|F_\phi(E_\theta(o_t), a_t)-E_\theta(o_{t+1})\|_2^2
+ \lambda_{\text{sig}}\mathcal L_{\text{SIGReg}}.
```

SIGReg 通过随机投影约束潜在分布，承担防止表示坍塌的职责。规划时，LeWM 使用动作序列
递归滚动潜在状态，并以预测终点和目标图像编码之间的欧氏距离作为 CEM cost。

因此，本项目不应把下列内容写成未经限定的新贡献：

- 首次防止 LeWM 表示坍塌；
- 首次让 LeWM 学出物理结构；
- 首次把 TD 或 value loss 用于 joint-embedding world model；
- 只凭成功率提升就声称 world model 本身得到改善。

### 2.2 LeWM 尚未保证什么

低一步潜在预测误差不自动保证：

- 多步自回归误差不会累积；
- 欧氏 latent distance 与真实可达时间或控制代价一致；
- 模型保留的是影响长期决策的状态因素，而不是容易预测但无关的视觉因素；
- 模型在数据覆盖不足、未见目标、未见 reward 或 OOD 物理参数下仍适合规划；
- 通过模型执行的 Bellman backup 与通过真实转移执行的 backup 一致。

最后一点是本方案的主要切入点。像素或 latent 预测目标对所有误差方向近似同等处理，
而控制任务只关心会改变未来可达性、价值和最优动作的误差方向。

## 3. 相关工作与新颖性边界

| 工作 | 与本方案直接相关的结论 | 留下的空间 |
| --- | --- | --- |
| LeWM | 无奖励、端到端、无需预训练的可滚动 JEPA world model；SIGReg 防坍塌 | 一步预测与欧氏目标距离不保证长期控制等价 |
| TD-MPC2 | 联合训练 latent consistency、reward、Q 和 policy，证明 TD 信号可以帮助潜在控制模型 | 依赖 reward，通常面向在线或给定任务，不是 reward-free 任意任务模型 |
| Temporal-Difference JEPA | 用 policy-conditioned successor prediction 和 TD bootstrap 学习 zero-shot RL 表示与策略 | 重点是 zero-shot policy，不提供 LeWM 式任意动作序列可滚动模型 |
| Value Equivalence | 模型可通过保持一组策略和价值函数的 Bellman 更新来定义，而不必重建全部观测细节 | 需要选择足够广的策略、任务和函数族，并处理近似误差 |
| Value-Guided JEPA Planning | 让表示距离近似 goal-reaching value 可以改善 JEPA 规划 | 直接 `LeWM + goal value loss` 已很接近该工作，创新空间有限 |
| RC-Aux | 在 LeWM 上加入多时域预测、可达性和 temporal negatives，直接处理“能预测但不能规划” | 本方案必须超越单一可达性目标，验证跨 reward 的 Bellman 等价 |
| Temporal-Distance JEPA | 学习有方向的轨迹时间代价和 rollout consistency，改善 LeWM 规划几何 | 不能把“学习时间距离”重新包装成本方案 |
| Fast LeWM | 使用 action-prefix 多时域预测缓解自回归误差并加快规划 | 与 Bellman 校准正交，可作为后续组合而非本方案贡献 |
| Sub-JEPA / SD-JEPA | 改进 SIGReg 的高斯先验或拆分潜在子空间 | 更直接针对坍塌与表示几何，说明本方案不宜主打 anti-collapse |

术语上必须区分两篇同名缩写工作：

- 本方案所借鉴的是 **Temporal-Difference JEPA**，即用于 zero-shot RL 的工作；
- 2026 年另一篇 **Temporal-Distance JEPA** 直接改进 LeWM 的时间距离，两者不是同一方法。

## 4. 可检验的研究假设

### H1：控制相关误差重加权

在有限数据和有限模型容量下，LeWM 不可能使所有潜在预测方向同时完美。Bellman
functionals 可以把优化容量集中到会改变长期回报、可达性或动作选择的预测误差上。

可证伪条件：加入 Bellman 校准后，held-out Bellman residual 不下降，或下降但固定
planner 下的长期控制没有改善。

### H2：跨任务而非单任务价值拟合

对足够多样的 reward directions 和 policy family 进行校准，比围绕一个训练 reward
优化 critic 更能保留 LeWM 的任务无关性，并能迁移到未见 reward 的线性组合。

可证伪条件：收益只出现在训练用伪任务，换目标或 reward 后消失。

### H3：局部预测与长期决策具有互补性

LeWM 的一步预测目标负责局部、可滚动的动力学结构；successor/Bellman 目标负责较长
时域的控制语义。联合目标应优于任一目标单独使用。

可证伪条件：TD-JEPA 式 policy 单独就达到全部收益，或 Bellman 目标使 LeWM 的多步
rollout 和物理 probe 明显退化。

### H4：分离状态表示与任务表示更稳定

状态编码器 `E` 应保留任务无关动力学，独立任务编码器 `\psi` 应描述 reward basis。
两者完全共享可能使单一价值几何覆盖通用物理信息。

可证伪条件：共享编码器在同等参数量下稳定地优于分离结构，并且没有泛化退化。

## 5. 候选路径比较

### 路径 A：只增加 TD/value head

冻结或基本保持 LeWM，在其 latent 上训练 critic、successor head 或 goal-conditioned
value，并在规划末端进行 value bootstrap。

- 优点：实现简单，最快判断 LeWM latent 是否含有价值信息；
- 缺点：成功率提升主要属于 planner augmentation，不能证明 RL 帮助了 world model；
- 定位：必须做的诊断 baseline，不建议作为最终方法。

### 路径 B：TD 梯度进入 encoder

让 TD/successor loss 更新 `E`，但不直接约束 `F`。

- 优点：可以形成 control-aware representation；
- 缺点：动力学只被 latent MSE 间接影响，表示与 predictor 可能共同漂移；
- 定位：关键消融，可以判断表示校准是否足够。

### 路径 C：Bellman functional 直接校准 dynamics

比较预测下一 latent 与真实下一 latent 在多个 Bellman functionals 下的响应，使
`F` 对控制相关函数近似保持等价。

- 优点：RL 信号直接进入 world model，研究主张最清楚；
- 缺点：目标非平稳，容易出现表示、critic、actor 共同迁移的退化解；
- 定位：推荐的核心方法。

### 路径 D：RL 主动采集更有信息的数据

使用 exploration policy 寻找高不确定性或高 surprise 转移，再训练 LeWM。

- 优点：RL 确实能改善物理覆盖；
- 缺点：改变了数据量和在线交互预算，不再是当前 reward-free offline 主协议；
- 定位：后续独立方向，不与本方案混合。

## 6. 推荐方法：BC-LeWM

### 6.1 数据与约束

训练集只包含离线、无外部奖励的转移：

```math
\mathcal D=\{(o_t,a_t,o_{t+1},d_t)\}.
```

主实验中，各方法必须使用相同数据、划分、观测预处理和动作空间。BC-LeWM 不得通过
额外环境交互获得优势，也不得在主 reward-free 协议中使用环境真实 reward。

### 6.2 Reward-free 任务基底

学习与状态编码器参数分离的任务投影。推荐的最小版本从状态编码器的 EMA 副本读取
latent，而不是再引入一个完整图像 backbone：

```math
u_t=\psi_\eta(\operatorname{sg}[\bar E(o_t)])\in\mathbb R^d.
```

从单位球或正交基中采样任务方向 `w`，定义伪奖励族：

```math
r_w(o_t)=w^\top u_t.
```

为了避免 `\psi` 与 value head 共同形成平凡解，应至少采用以下约束：

- `\psi` 使用 whitening、orthogonality 或方差约束；
- Bellman target 使用 target network 和 stop-gradient；
- `\psi` 与 `E` 分离，或至少从 `E` 的 EMA 副本读取输入；
- 定期测量 task basis 的有效秩、协方差谱和不同 `w` 的 reward 相关性。

“任意 reward”只能在明确的函数张成假设下成立。正文应称为“task-family”或
“reward-basis generalization”，不能直接声称覆盖所有 reward。

### 6.3 Policy-conditioned successor 学习

学习条件策略 `\pi_\xi(a\mid z,w)` 和 successor representation
`M_\omega(z,a,w)`。在真实离线转移上使用：

```math
\mathcal L_{\text{SF}}
=\left\|
M_\omega(z_t,a_t,w)
-\operatorname{sg}\left[
u_{t+1}+\gamma M_{\bar\omega}(z_{t+1},a',w)
\right]
\right\|_2^2,
```

其中 `a' \sim \pi_\xi(\cdot\mid z_{t+1},w)`，`\bar\omega` 为 EMA target。对应伪任务
的 action-value 可以写成：

```math
Q_w(z,a)=w^\top M_\omega(z,a,w).
```

由于这是离线学习，策略必须受到数据支持约束。候选措施包括 behavior cloning 正则、
动作 KL、advantage-weighted regression 或保守 Q 正则。第一版不应同时引入多种复杂
offline RL 技巧，应先选择包内公开 baseline 最接近、可审计的一种。

### 6.4 Bellman calibration

记 LeWM 预测的下一状态为：

```math
\hat z_{t+1}=F_\phi(z_t,a_t), \qquad z_{t+1}=E_\theta(o_{t+1}).
```

对一组任务方向和策略，定义 Bellman functional：

```math
G_{w,\pi}(z)
=\rho_w(z)+\gamma\,\mathbb E_{a\sim\pi_w(\cdot\mid z)}
[Q_w(z,a)],
```

其中 `\rho_w(z)=w^\top\bar\psi(z)` 是由冻结或 EMA 任务投影得到的伪奖励响应，因而
既可以作用于真实 latent，也可以作用于模型预测 latent。核心校准目标为：

```math
\mathcal L_{\text{BC}}
=\mathbb E_{w,\pi}
\left[
\left\|
G_{w,\pi}(\hat z_{t+1})
-\operatorname{sg}G_{w,\pi}(z_{t+1})
\right\|_2^2
\right].
```

它要求预测转移与真实转移对选定 Bellman 函数族产生近似相同的后果。严格来说，这只
是在采样的任务、策略和函数集合上实现近似 value equivalence，而不是全体 MDP 意义
下的 universal equivalence。

### 6.5 总目标

```math
\mathcal L
=\mathcal L_{\text{LeWM}}
+\alpha\mathcal L_{\text{SF}}
+\beta\mathcal L_{\text{BC}}
+\lambda_{\text{offline}}\mathcal L_{\text{support}}.
```

第一版 MVP 应控制变量：先只加入一个 successor objective 和一个 Bellman calibration，
不要同时混入多时域 transformer、对比学习、重建 decoder 或新 planner。

### 6.6 推荐训练顺序

1. **LeWM warm-up**：只训练原始预测和 SIGReg，建立稳定的局部动力学；
2. **functional fitting**：冻结或 EMA 化 `E/F`，在真实转移上拟合 task basis、successor
   head 和受数据约束的 policy；
3. **model calibration**：冻结 Bellman target，使用 `L_BC` 更新 `E/F`；
4. **alternating update**：交替更新 functional 和 world model，避免所有模块同时追逐；
5. **joint fine-tuning**：只有前三阶段稳定后，才小权重联合训练并观察梯度冲突。

训练日志需要分别记录 `L_LeWM`、`L_SF`、`L_BC`、SIGReg、latent effective rank、
Bellman target 漂移、actor 数据支持程度以及不同损失在 `E/F` 上的梯度余弦相似度。

### 6.7 测试时使用方式

应明确区分两个版本：

- **BC-LeWM dynamics only**：评测时仍用原始 LeWM 欧氏目标 cost 和完全相同的 CEM，
  用于证明训练后的 world model 本身更适合规划；
- **BC-LeWM + value bootstrap**：CEM 终点再加 successor/value terminal cost，用于测量
  完整系统上限，但不能把这部分收益归因于 dynamics。

## 7. 实验协议

### 7.1 协议 A：Reward-free arbitrary-goal planning

这是与 LeWM、PLDM、DINO-WM 最直接的公平协议：

- 训练时没有环境 reward；
- 使用相同离线数据、训练 split 和训练 epoch；
- 固定 CEM candidates、elites、iterations、horizon、action block 和 replanning 周期；
- 测试目标由目标观测给定；
- 主结果使用相同 planner cost，value bootstrap 只作为单独结果。

### 7.2 协议 B：Zero-shot reward transfer

这是与 Temporal-Difference JEPA、FB/HILP 类方法比较的协议：

- world model 和 representation 训练期间无外部 reward；
- 测试时提供从未见过的 reward function；
- 各方法使用完全相同、规模很小的 reward-labelled inference set 拟合任务方向；
- 明确报告 reward inference 样本数、是否需要环境交互以及规划/策略计算预算；
- 分别测试 reward 位于训练 task basis 的 span 内和 span 外的情形。

协议 A 与 B 可以共享模型，但不能把两个协议的 baseline 数字放进同一排名表。

### 7.3 协议 C：Reward-labelled control

TD-MPC2 需要真实 reward/Q 学习，应放在独立协议中。只有为所有方法定义相同 reward
标注、在线交互和训练预算后，才可以与 BC-LeWM 进行系统级比较。它不能直接替代协议
A 中的 reward-free baseline。

## 8. Baseline 与消融矩阵

### 8.1 必需 baseline

- LeWM；
- PLDM；
- DINO-WM pixels-only；
- Temporal-Difference JEPA 或其公开 zero-shot RL 结果（协议 B）；
- value-guided JEPA、RC-Aux、Temporal-Distance JEPA 中可公平复现的方法；
- TD-MPC2（仅协议 C）。

本项目必须优先使用 `stable-worldmodel==0.1.1` 的公开 API 和已发布实现，不复制或
维护 baseline 源码分叉。尚未接入统一协议的方法只能作为文献参照，不能伪装成已复现
结果。

### 8.2 归因消融

| 编号 | 设置 | 回答的问题 |
| --- | --- | --- |
| A0 | 原始 LeWM | 基准线 |
| A1 | LeWM + detached SF/value head | latent 本身是否已支持价值学习；收益是否只来自 planner |
| A2 | TD/SF 梯度进入 encoder，不进入 dynamics | control-aware representation 是否足够 |
| A3 | `L_BC` 只进入 dynamics | Bellman calibration 是否真的改善转移模型 |
| A4 | `L_BC` 同时进入 encoder 和 dynamics | 完整耦合是否优于受限耦合 |
| A5 | 完整 BC-LeWM，但评测不使用 value head | world model 改进的核心证据 |
| A6 | 完整 BC-LeWM + value bootstrap | 完整系统上限 |
| A7 | TD-JEPA 式 policy，不使用 LeWM rollout | 显式 world model 是否仍有必要 |
| A8 | 随机等参数辅助 head | 收益是否只是参数量或额外正则造成 |

还应消融 task directions 数量、共享/分离 encoder、warm-up 长度、`alpha/beta`、target
EMA、策略数据支持约束和 Bellman horizon。一次实验只改变一个关键因素。

## 9. 指标与诊断

### 9.1 World model 指标

- one-step latent prediction loss；
- 5/10/20-step open-loop rollout degradation；
- action sensitivity：改变动作后预测是否产生可区分且合理的变化；
- 真实转移与模型转移上的 held-out Bellman residual；
- rollout 中 Bellman residual 随 horizon 的增长曲线；
- 参数量、训练时间、推理时间和显存。

### 9.2 表示指标

- latent covariance spectrum、effective rank、SIGReg 统计；
- agent/object 位置、角度、速度等 physical probes；
- reachability、temporal distance、controllability probe；
- task basis 的秩、正交性和不同任务方向覆盖度；
- 训练 task directions 与未见 directions 上的 value linear probe。

### 9.3 控制指标

- `world.evaluate(...)` 返回的 success rate 和逐 episode success；
- return、episode length、规划时间和候选动作预算；
- IID goal、unseen goal、unseen reward；
- OOD visual FoV、OOD physics FoV；
- 低覆盖、次优数据和需要 trajectory stitching 的场景。

### 9.4 统计口径

- 正式结果至少 3 个独立训练 seeds；
- 每个 checkpoint 使用固定且相同的评测 episode 集合；
- 保存逐 episode 原始结果，而不是只保存均值；
- 报告均值、标准差、置信区间，并尽可能报告 paired difference；
- 超参数选择只使用 validation tasks，不能查看测试 reward 后调参。

## 10. 任务选择

现有四任务中：

- PushT 上 LeWM 已达到约 96%，容易出现天花板效应，适合验证不退化，不适合作为唯一
  提升证据；
- Two-Room 多个方法已接近饱和，原始版本区分度不足；
- Reacher 尚有一定区分度，可用于检查连续控制与动力学表示；
- Cube 上 LeWM 明显低于 DINO-WM，且随机成功率较高，需要同时看绝对与相对提升。

建议首批研究任务为：

1. **Reacher**：快速迭代 TD/Bellman 稳定性；
2. **Cube**：检查视觉、控制语义和未见目标泛化；
3. **Deceptive / OOD navigation**：构造欧氏距离与可达性不一致的新布局；
4. **PushT**：作为不退化和接触动力学验证，而非主要刷分环境。

若 zero-shot reward transfer 是论文主张，还需增加带多种 reward 的像素控制任务。新增
环境之前必须先确认 Stable World Model 的公开支持和统一评测接口，不能为了方法临时
维护另一套不可比框架。

## 11. 最有判别力的主实验

在完全相同的 reward-free 数据上训练 LeWM 和 BC-LeWM。训练结束后冻结全部模型，
再提供训练期间未见的目标或 reward，并给各方法完全相同的少量 reward inference
样本。使用相同 CEM、相同候选动作和相同计算预算评测。

只有同时观察到以下结果，才支持核心假设：

1. BC-LeWM 在 held-out task/policy functionals 上具有更低的 model-vs-data Bellman
   residual；
2. 不使用新 value terminal cost 时，BC-LeWM 仍提高长期规划成功率；
3. 提升在未见目标、未见 reward 或 OOD 场景中存在；
4. LeWM 的一步预测、长时 rollout 和物理 probes 没有被明显破坏；
5. detached value head、参数量匹配和 planner-only baseline 无法解释全部收益。

这个实验比单纯比较最终成功率更重要，因为它直接区分“模型被 RL 改善”和“模型不变，
只是外挂了一个更好的 critic”。

## 12. 失败判据与结果解释

以下结果不支持“RL 改善 LeWM world model”的主张：

- 只有启用 value bootstrap 时成功率才提升；
- 固定相同 planner 和 cost 后增益消失；
- Bellman residual 下降，但 open-loop rollout 或物理 probes 大幅恶化；
- 收益仅存在于训练使用的一个 reward；
- 方法必须使用额外真实 reward 或更多环境交互；
- 提升来自更多参数、更多训练 step 或更大的规划预算；
- 不同 seeds 高度不稳定，平均增益由单个 seed 主导。

对应的诚实结论应分别是：

- planner/value augmentation，而不是 world model improvement；
- task-specific representation learning，而不是 reward-free generalization；
- prediction/control trade-off，而不是全面更好的表示；
- 在当前 task basis 或优化方式下，研究假设未成立。

## 13. 主要风险与缓解措施

### 13.1 非平稳目标与梯度冲突

风险：`E/F/psi/M/pi` 同时更新会让 TD target 快速漂移，SIGReg 和 Bellman loss 也可能
推动 encoder 朝相反方向变化。

缓解：warm-up、EMA target、stop-gradient、交替训练、低权重引入 `L_BC`、记录每个
损失对共享参数的梯度夹角。只有观察到冲突后再考虑 gradient surgery。

### 13.2 Task basis 的平凡解或循环定义

风险：伪 reward、value 和 world model 由同一 latent 同时定义，可能共同改变坐标系，
使损失下降但没有获得外部可解释的控制信息。

缓解：分离 `E` 与 `psi`，使用 EMA/frozen teacher、whitening 和正交约束；使用未参与
训练的环境状态 probes 和 held-out reward 检验。

### 13.3 离线策略的分布外动作

风险：conditioned actor 选择数据集未覆盖的动作，critic 和 model 相互强化虚假的高
价值预测。

缓解：行为约束、保守正则、action support 指标；先在 behavior-policy 或 future-goal
relabeling 下验证，再逐步增加策略自由度。

### 13.4 价值等价牺牲通用预测

风险：value-equivalent model 可以有意忽略与当前函数族无关的信息，导致换任务后失效。

缓解：始终保留 LeWM prediction + SIGReg 主目标，扩大 task/policy basis，并在未见
reward、OOD physics 和物理 probes 上检查信息保留。

### 13.5 新颖性被近期工作覆盖

风险：近期 LeWM 改进已经覆盖 value distance、reachability、temporal distance 和
multi-horizon prediction。

缓解：把贡献严格限定为“reward-free task/policy family 上的 model-vs-data Bellman
functional calibration”，并在写作和实验中正面对比最接近的方法。

## 14. 实施路线与决策门

### 阶段 0：合规 baseline

- 通过 `stable-worldmodel[all]==0.1.1` 的公开 API 复现 LeWM checkpoint evaluation；
- 固定数据、CEM、评测 episode 和记录格式；
- 在 baseline 尚未可靠复现前，不启动大规模方法训练。

通过条件：至少一个正式任务上的 LeWM 结果和运行链路可审计、可恢复。

### 阶段 1：Frozen-latent feasibility probe

- 冻结 LeWM，训练 task basis、successor/value heads；
- 测量 value probe、Bellman residual 和 zero-shot reward fitting；
- 不修改 world model，不声称方法提升。

停止条件：LeWM latent 连最基本的 held-out TD/value probe 都无法稳定学习，或 task
basis 持续坍塌。此时先修正任务构造，不进入联合训练。

### 阶段 2：最小 Bellman calibration

- 先做 A1/A2/A3 消融；
- 只在 Reacher 或一个小任务运行少量 steps；
- 检查 loss、梯度、EMA、checkpoint resume 和多步 rollout。

通过条件：`L_BC` 可重复下降，held-out Bellman residual 改善，同时原始 LeWM 指标
没有明显退化。

### 阶段 3：固定规划器的控制验证

- 使用相同 CEM 和原始目标 cost 比较 A0/A2/A3/A5；
- 运行至少 3 seeds；
- 加入 Cube、OOD 目标或 deceptive layout。

通过条件：不依赖新 value terminal cost 仍有稳定控制收益。

### 阶段 4：Zero-shot reward transfer

- 预注册训练 task basis、held-out reward 和 reward inference 样本数；
- 与 Temporal-Difference JEPA 类方法进行独立协议比较；
- 再测试完整 BC-LeWM + value bootstrap 的系统上限。

## 15. 与当前仓库的集成原则

实现时保持三层结构：

- `stable_worldmodel` 负责环境、数据、baseline、规划和统一评测；
- 本仓库只实现 BC-LeWM 的 task basis、successor/Bellman objective、训练组装和轻量
  public-API adapter；
- 配置显式记录 baseline、方法、随机种子、任务基底和 loss 权重。

不得修改已安装的 `stable_worldmodel` 包，不复制上游 LeWM/TD-MPC2 实现，不把
baseline 与 proposed method 混在同一模型类中。建议在研究假设通过阶段 1 决策门后，
再创建 `src/tdwm/methods/` 等实现目录，避免在方法尚未定型时提前搭建空框架。

## 16. 论文叙事建议

### 不推荐的标题式表述

- RL Prevents Representation Collapse in LeWM；
- Making LeWM Physically Meaningful；
- Combining TD-JEPA and LeWM；
- A Better LeWM。

这些表述要么与 LeWM 已有主张冲突，要么只是模块拼接，没有指出新的科学问题。

### 推荐的核心表述

> Predictive latent dynamics are not necessarily decision-equivalent. We investigate whether
> reward-free Bellman functionals learned over a task and policy family can calibrate an
> end-to-end JEPA world model for long-horizon control without sacrificing its task-agnostic
> predictive structure.

中文可以概括为：

> 可预测不等于对决策等价。BC-LeWM 使用无奖励任务族上的 Bellman functionals 校准
> LeWM 转移，使有限模型容量优先保留影响长期控制的动力学因素。

### 潜在贡献点

若实验成立，可主张：

1. 一种在无真实 reward 离线数据上构造 task/policy Bellman functionals 的方法；
2. 一种直接比较模型转移与数据转移 Bellman 响应的 LeWM 校准目标；
3. 将 world-model improvement 与 planner/value augmentation 分离的评测协议；
4. 关于预测精度、物理表示、Bellman 等价和 OOD 控制之间关系的实证分析。

## 17. 尚待决定的问题

1. 任务特征 `psi` 使用独立图像 encoder，还是使用 `E` 的 EMA latent 加小型 projection？
2. 第一版 task basis 使用随机线性 reward、future-goal reward，还是两者都做？
3. Successor head 输出向量 successor features，还是直接输出一组标量 value functionals？
4. Policy family 使用显式 conditioned actor，还是先使用行为策略和数据内 action proposals？
5. Bellman calibration 只约束一步预测，还是对 3/5-step model rollout 同时约束？
6. 主要论文协议应以 arbitrary-goal planning 为主，还是以 zero-shot reward transfer 为主？
7. 哪个任务能同时避免 PushT/Two-Room 饱和，并由 Stable World Model 公开接口支持？

当前推荐的最小选择是：`EMA(E) + orthogonal task projection`、标量 Bellman functional
ensemble、受 behavior regularization 的 conditioned policy，以及先做 one-step calibration。
在该版本证明有效以前，不引入多步架构或新 planner。

## 18. 下一步具体产物

在开始实现前，应依次形成：

1. 一页 pre-registration：H1-H4、主指标、停止条件和 held-out tasks；
2. 一份 Stable World Model `0.1.1` 公共接口核对记录；
3. LeWM baseline 的合规 checkpoint evaluation；
4. frozen-latent TD/SF feasibility probe 的最小实验配置；
5. 通过阶段 1 后再编写 BC-LeWM 方法代码和完整消融配置。

## 参考资料

- LeWM：<https://arxiv.org/abs/2603.19312>
- LeWM 官方实现：<https://github.com/lucas-maes/le-wm>
- Temporal-Difference JEPA：<https://arxiv.org/abs/2510.00739>
- Temporal-Difference JEPA 官方实现：<https://github.com/facebookresearch/td_jepa>
- TD-MPC2：<https://arxiv.org/abs/2310.16828>
- Value Equivalence Principle：<https://arxiv.org/abs/2011.03506>
- Proper Value Equivalence：<https://arxiv.org/abs/2106.10316>
- Value-Aware Model Learning：<https://proceedings.mlr.press/v54/farahmand17a.html>
- Calibrated Value-Aware Model Learning：<https://arxiv.org/abs/2505.22772>
- Value-Guided Action Planning with JEPA World Models：<https://arxiv.org/abs/2601.00844>
- Predictive but Not Plannable / RC-Aux：<https://arxiv.org/abs/2605.07278>
- Temporal-Distance JEPA：<https://arxiv.org/abs/2607.25337>
- Fast LeWM：<https://arxiv.org/abs/2606.26217>
- Temporal Straightening：<https://arxiv.org/abs/2603.12231>
- Sub-JEPA：<https://arxiv.org/abs/2605.09241>
- SD-JEPA：<https://arxiv.org/abs/2605.31111>
- AdaJEPA：<https://arxiv.org/abs/2606.32026>
- PiJEPA：<https://arxiv.org/abs/2603.25981>
- Stable World Model 文档：<https://galilai-group.github.io/stable-worldmodel/>
- Stable World Model `0.1.1`：<https://pypi.org/project/stable-worldmodel/0.1.1/>

## 当前结论

BC-LeWM 目前是一个**可证伪且边界相对清楚的研究提案**，不是已经成立的方法。最先
应该验证的不是最终成功率，而是 frozen LeWM latent 上能否稳定学习多样的 reward-free
Bellman functionals，以及这些 functionals 是否能测出一步 latent MSE 没有反映的长期
控制误差。只有这个信号存在，继续让 RL 梯度进入 LeWM 才有充分依据。

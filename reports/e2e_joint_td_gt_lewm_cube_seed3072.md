# E2E Joint TD-GT-LeWM Cube 完整训练、评测与偏差审计

训练与评测完成：2026-08-22（Asia/Shanghai）

记录状态：单个训练 seed、单个评测 seed 的完整端到端工程运行。该运行从随机初始化和
Cube 原始图像开始，确实联合优化了 LeWM 与 GoalTailValue；但实际训练目标、采样和评测
场景没有严格实现预期的 residual planner-tail formulation。因此本记录可以判定该工程配置
没有超过 baseline，不能据此判定理论 formulation 成立或被否定。

## 本次实际训练的方法

本次工程方法命名为 **E2E Joint TD-GT-LeWM**。它不加载 LeWM checkpoint 或 latent
cache，在同一个训练图和优化器中更新：

```text
encoder + projector + predictor + action_encoder + pred_proj + GoalTailValue
```

LeWM 保留 prediction MSE 和 SIGReg。GoalTailValue 接收 5-step 可微 LeWM rollout 的
terminal history，并使用一步 TD 目标；tail loss 继续通过 imagined rollout 反传到 LeWM。
训练共执行 10 epochs、127,960 次 optimizer update，不是冻结 LeWM 后只训练 value head
的短时诊断。

## 完整训练结果

按预先配置的 minimum validation joint loss 规则选择 epoch 9：

| 指标 | epoch 9 |
| --- | ---: |
| validation prediction MSE | 0.008731 |
| validation 5-step terminal rollout MSE | 0.019560 |
| validation TD tail MSE | 0.003121 |
| 训练耗时 | 约 9 小时 22 分钟 |
| optimizer updates | 127,960 |
| accelerator | NVIDIA GeForce RTX 3090 |
| 峰值显存 | 约 10.5 GB |

训练后通过 109 项测试。以上 loss 只能验证训练链路和相应监督目标可优化，不能直接推出
CEM candidate ranking 或 Cube control performance。

## 受控 CEM 结果

三项评测使用相同的 50 个 start--goal pair、planning seed、300 candidates、30 iterations
和原始 LeWM CEM 参数：

| 评测模型 | 成功数 | success rate |
| --- | ---: | ---: |
| 原始 LeWM baseline | 36 / 50 | 72% |
| E2E epoch 9 LeWM，不加入 tail | 31 / 50 | 62% |
| E2E epoch 9 LeWM + GoalTailValue | 28 / 50 | 56% |

因此该工程运行的观测差异可拆成：联合训练后的 world model 相对 baseline 下降 10 个
百分点；将本次训练的 tail 接入该 world model 后再下降 6 个百分点。

baseline 与完整 E2E 方法的配对结果为 both-success 24、baseline-only 12、E2E-only 4、
neither 10。McNemar exact `p=0.0768`；paired bootstrap 的 success-rate difference 为
`-0.16`，95% interval 为 `[-0.30, -0.02]`。这是单一训练 seed 和单一 50-episode
评测样本的结果，不构成跨 seed 的方法优劣结论。

## 工程实现与理论 formulation 的偏差

### 1. 成功 terminal 的零边界没有被训练

正式评测使用 goal offset 25、CEM horizon 5 和 action block 5。因此每次候选计划正好
覆盖数据轨迹中从 start 到 goal 的 25 个环境 step。对已经到达 goal 的候选而言，正确的
residual tail 必须满足：

\[
V(h_{t+H},z_g)=V(h_{t+H},z_{t+H})=0.
\]

实际训练只采样正 future offset `1..16`，没有 offset 0，也没有显式边界约束
`V(h,z_current)=0`。在 65,536 个离线诊断 goal pair 上，实际得到：

| 边界输入 | value mean | value std |
| --- | ---: | ---: |
| true terminal history，goal=current latent | 0.65239 | 0.27432 |
| predicted terminal history，goal=predicted terminal latent | 0.65390 | 0.27426 |

planner adapter 为匹配 LeWM 的 summed latent cost，将 mean-dimensional value 乘以 latent
dimension 192。因而本应为零的边界项平均变为约 `0.654 * 192 = 125.6`，并携带约
`0.274 * 192 = 52.7` 的标准差。这个换算在单位上符合当前 adapter 定义，但暴露并放大了
value 的边界校准错误。删除 tail 后 success rate 从 56% 回升到 62%，与该诊断一致。

未到达 goal 的候选仍可能存在 horizon 之后的 recovery cost，因此不能简单说整个评测中
“没有 tail”。但当前 `o25` 协议让 baseline candidate 已覆盖数据中的 nominal start--goal
horizon，而且没有单独隔离 horizon 外信息带来的作用。它不是对 boundary-correct residual
tail 的干净检验。

### 2. pooled Spearman 高估了 planner ranking

同一离线诊断得到 MC target MSE `0.01058`、MAE `0.06312` 和 pooled Spearman
`0.94421`。但 pooled 指标混合了不同 future offset，主要奖励了“更远 goal 通常具有更大
累计 cost”这一容易的排序信号。

固定 offset 后，offset 2--6 的 Spearman 依次只有 `0.3360`、`0.2642`、`0.3036`、
`0.3731` 和 `0.4194`；offset 1 的 MC target 恒为零，相关系数未定义。CEM 真正需要的是
同一 goal 和 horizon 下不同 candidate terminal history 的排序，而不是跨 offset 排序。
因此 pooled Spearman 不能作为本方法已学会 planner-relevant tail 的证据。

### 3. SIGReg 的有效 batch 没有保持独立样本数

原始 LeWM baseline 使用 128 条独立 clip。E2E 配置为了控制原始图像训练的显存，将
16 条 clip 各展开为 8 个高度重叠 local prediction window，并把它记作 SIGReg effective
batch size 128。两者的 tensor 数量相同，但独立轨迹数和样本多样性不同，不能视为严格
等价的 LeWM 训练协议。

与 world-only CEM 从 72% 降至 62% 一致，E2E epoch 9 的 validation prediction MSE
为 `0.008731`，原始 LeWM epoch 10 为 `0.005418`。由于两次 validation clip 布局并不
完全相同，这个 MSE 对比只作为支持性诊断，不作为精确的独立因果估计。

### 4. 当前工程 TD target 与拟检验的 planner-tail 语义没有严格对齐

当前 TD target 来自离线 scripted behavior trajectory 的 hindsight future goal。它直接监督
的是数据行为继续执行时的累计 latent cost。若预期 formulation 中的 tail 表示候选 terminal
state 之后在最优、planner-conditioned 或其他明确 continuation policy 下的 cost-to-go，
则当前 target 并没有实现这个语义。

当前 scalar `V(h,g)` 没有 continuation action 或 policy 条件；训练 terminal history 来自
behavior action rollout，推理 terminal history 来自优化后的反事实 CEM candidate。因而这里
可能同时存在目标语义差异和分布偏移。CEM 还可能主动选择 value 低估的候选。这是由当前
工程训练与规划定义推导出的机制风险，不是对理论 formulation 本身的反例；本次 50-episode
结果也不能单独区分它和其他偏差的贡献。

### 5. 联合 TD 表征目标是非平稳的

本次训练只有 target value network，没有 target encoder 或 target world model。TD goal、
immediate cost 和 bootstrap history 均依赖持续更新的在线 latent geometry，同时 tail loss
反传到 encoder 与 predictor。因此网络可以改善当前 Bellman self-consistency，却不一定
保留对 CEM 有效的目标距离和 dynamics。该机制可能参与了 world-model 退化，但尚未通过
单变量消融确认。

## 可以和不可以得出的结论

本次实验可以确认：

1. E2E Joint TD-GT-LeWM 的完整训练链路已经真实运行，并非加载已有 LeWM 后只训练小型
   value head。
2. 这一具体代码、训练配置和 `o25` CEM 评测组合没有超过原始 LeWM baseline。
3. 性能下降同时出现在联合训练后的 world model 和 tail planner term，不能只归因于一个
   scalar 权重。
4. 当前边界监督、评测 horizon、独立 batch 组成和 value 语义与预期理论 formulation
   存在实质偏差。

本次实验不可以确认：

1. 预期的 residual GoalTail formulation 已被严格实现并被否定。
2. 在 goal 位于 planner horizon 之外、边界正确、LeWM 训练协议等价时，tail 仍然无效。
3. TD、joint representation learning 或 planner-conditioned long-horizon value 本身无效。
4. 任一方法优于另一方法；正式方法比较仍需要修正协议后的多 seed 受控实验。

因此准确结论是：**本次工程实例是负结果，并暴露了实现与实验协议的关键偏差；它没有
构成对预期精确 formulation 的有效证伪。**

## 修正后的下一次检验条件

下一次正式运行前应先满足以下必要条件，而不是原样重复本配置：

1. 加入 offset 0 样本或等价的严格边界约束 `V(h,z_current)=0`。
2. 令 goal 严格位于 CEM planning coverage 之外，例如保持 25-step planning coverage，
   使用更远的 goal offset。
3. 使用真正独立的 128 条 trajectory/clip 保持 SIGReg 协议，不能用重叠窗口数量代替独立
   样本数。
4. 报告固定 goal、固定 offset 的 candidate-ranking 指标，并直接评估 planner candidates；
   pooled future-offset Spearman 只保留为辅助指标。
5. 明确定义 tail 是 behavior value、optimal value、policy-conditioned value 还是
   action-conditioned Q，并让训练 target 与 CEM 中的使用语义一致。
6. 通过 stop-gradient、target encoder/world model、分阶段 warmup 或对应消融，隔离 TD
   representation drift 对 LeWM 的影响。

这些是使工程实例更接近理论 formulation 的条件，不预先保证修正后一定提高性能。

## 审计信息

| 项目 | 值 |
| --- | --- |
| Stable World Model | `stable-worldmodel[all]==0.1.1` |
| training seed | 3072 |
| planning seed | 42 |
| selected checkpoint | epoch 9 |
| LeWM checkpoint SHA-256 | `fb86225b2ae3bbeb3e94472be1858c351e63a73760081b6d885b213c9c83e3b0` |
| full joint checkpoint SHA-256 | `97dbe51eba84cf04f977d862e10a16e3d59a791e9b392c3a24dc9edb970a41cf` |
| training implementation commit | `5590571` |
| evaluation implementation commit | `d108fee` |
| locked evaluation protocol commit | `4890305` |

正式训练配置为
[`configs/experiment/e2e_joint_td_gt_lewm_cube_train.yaml`](../configs/experiment/e2e_joint_td_gt_lewm_cube_train.yaml)，
正式评测配置为
[`configs/experiment/e2e_joint_td_gt_lewm_cube_seed3072_o25.yaml`](../configs/experiment/e2e_joint_td_gt_lewm_cube_seed3072_o25.yaml)。
world-only 诊断结果保留在外部运行目录：

```text
/home/yabin/tdwm/outputs/diagnostics/e2e_world_only_seed3072_epoch9_o25/results.json
```

checkpoint、原始日志和大型结果均保留在外部 artifact 目录，不提交到 GitHub。边界和
fixed-offset 指标属于正式评测后的 post-hoc diagnosis，并非预先锁定的主结果。

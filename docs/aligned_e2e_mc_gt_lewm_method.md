# Aligned E2E MC-GT-LeWM 方法说明

## 一句话概括

Aligned E2E MC-GT-LeWM 是一个 **LeWM latent world model + goal-conditioned
long-horizon cost** 的联合训练方法：它从 Cube 原始图像和随机初始化开始，同时学习短期
latent dynamics 与一个基于 Monte-Carlo future targets 的目标尾部代价（MC Goal Tail），
再把两者接入同一个 CEM planner。

本文描述当前仓库中的实际实现，不把它扩展解释成尚未验证的通用方法。

## 名称含义

| 名称 | 含义 |
| --- | --- |
| **LeWM** | latent world model。编码观测和动作，预测未来 latent。 |
| **MC-GT** | Monte-Carlo Goal Tail。用离线轨迹中的未来 latent 计算到目标的折扣累计代价，不做 TD bootstrap。 |
| **E2E** | end-to-end。从原始图像随机初始化联合训练 world model 和 tail value，不使用冻结的 LeWM checkpoint 或 latent cache。 |
| **Aligned** | tail 使用与当前 LeWM 对齐的 latent 坐标、EMA target world model 和结构化零边界，避免把不同表示空间的 value 直接混用。 |

## 为什么需要 tail value

普通 LeWM CEM 通常主要依据候选动作 rollout 的终点 latent 与目标 latent 的距离。对于
较长的 start--goal 间隔，这个终点距离可能过于短视，且对中间轨迹质量不敏感。

本方法让一个 goal-conditioned value head 估计：

> 给定当前 latent 历史和目标 latent，沿离线数据行为继续走一段时间的累计 latent goal cost。

因此 planner 的 cost 不只看一次终点距离，也可以利用训练过的长期目标代价。

## 模型结构

### 1. LeWM world model

在线 world model 从原始 Cube 图像得到 latent，并基于历史 latent 和动作预测未来 latent。
它保留原 LeWM 的两项训练目标：

```text
L_world = next-latent prediction MSE + 0.09 * SIGReg
```

正式配置使用 3 个 history frames、5-step model rollout、latent embedding size 192。

### 2. Boundary-anchored MC Goal Tail

tail value 接收：

```text
V(history, goal) -> scalar cost
```

其中 `history` 包含预测 rollout 末端的 latent history 和此前动作，`goal` 是未来 latent。
它不是普通的自由输出 MLP，而是通过共享 scalar potential 构造：

```text
V(h, g) = [phi(h, g) - phi(h, z_current)]^2
```

所以它有两个重要性质：

1. `V >= 0`；
2. `V(h, z_current) = 0` 是结构上严格成立的边界条件，而不是依赖额外 penalty 学出来的近似值。

### 3. Monte-Carlo targets

对每个 tail 样本，先让在线 LeWM 对前 5 个动作进行可微 rollout，得到预测的 terminal
history。然后用 EMA world model 编码真实离线序列的后续 latent，构造未来 offset 1--16
的目标。

对 offset `k`，target 是从未来第 1 步到第 `k` 步的折扣 latent goal distance：

```text
y_k = (1 - gamma) * sum_{j=1..k} gamma^(j-1) * d(z_{T+j}, z_{T+k})
```

当前 `gamma = 0.95`，目标 offset 在 1--16 间均匀采样，continuation policy 是离线数据
中的 behavior continuation。该 target 是直接的 supervised MC target，不使用 value
bootstrap。

## “Aligned”具体对齐了什么

它不是简单地在已经训练好的 LeWM 后面外挂一个 value head，而是同时处理以下对齐问题：

- **表示对齐**：target latent 来自在线 LeWM 的 EMA 副本，和当前模型保持同一表示演化轨迹；
- **训练路径对齐**：tail 的输入是 online LeWM 预测 rollout 得到的 terminal history，而不是只在真实 latent 上训练；
- **目标边界对齐**：当前状态作为自己的 goal 时，tail cost 精确为零；
- **数据视图对齐**：同一次 optimizer update 使用 128 个独立短 clip 训练原始 LeWM loss，另用 16 个 long clip 训练 MC tail，避免用同一个短 clip 假装提供长期监督。

EMA target world model 的 decay 为 `0.995`。tail loss 在训练开始时 warm up，并把传回 world
model rollout 的梯度缩放为 `0.1`；value head 本身仍正常更新。

联合目标可以简写为：

```text
L_joint = L_world + warmup(t) * L_MC_tail
```

两项 loss 在同一个 optimizer update 中更新同一个 online LeWM；EMA model 只作为无梯度
target encoder。

## 规划时如何使用

CEM solver 和 LeWM 的基础设置保持不变。候选动作 rollout 后，使用：

```text
planning cost = terminal latent-goal distance
              + boundary-anchored MC tail cost
```

当前 Cube 正式协议为 horizon 5、300 candidates、30 iterations、30 elites，每 5 个环境
step 重规划。

## 与冻结版 MC-GT-LeWM 的区别

| 项目 | 冻结版 MC-GT-LeWM | Aligned E2E MC-GT-LeWM |
| --- | --- | --- |
| LeWM | 已训练并冻结 | 从随机初始化联合训练 |
| 输入 | 缓存 latent | Cube 原始图像 |
| tail target | 固定 world model latent | EMA world model latent |
| tail 输入 | 主要是真实/缓存 latent history | online model rollout 的 predicted terminal history |
| 零边界 | 普通 value 形式 | 结构化 exact zero boundary |
| 训练关系 | 先后分离 | world loss 与 tail loss 同步更新 |

## 当前实验结果与证据边界

在固定 training seed 3072 的 Cube 评测中：

- 历史单次 planning selection（50 episodes）：`31/50 = 62%`；
- 后续 6 组 matched selections（共 300 episodes），完整方法（Aligned world + anchored tail）：`167/300 = 55.67%`；
- 同一批次的 Original LeWM world-only：`153/300 = 51.0%`；
- 去掉 inference-time tail 的 Aligned world-only：`168/300 = 56.0%`。

这些结果只有一个 training seed。当前证据更支持“长期监督改善了训练出的 world model”，
而不是“推理时追加 tail 一定带来稳定收益”，暂不能据此声称统计意义上的方法优越性。

## 仓库中的实现入口

- 方法配置：[`../configs/methods/aligned_e2e_mc_gt_lewm.yaml`](../configs/methods/aligned_e2e_mc_gt_lewm.yaml)
- 训练配置：[`../configs/experiment/aligned_e2e_mc_gt_lewm_cube_train.yaml`](../configs/experiment/aligned_e2e_mc_gt_lewm_cube_train.yaml)
- 训练实现：[`../src/tdwm/training/aligned_e2e_mc_gt_lewm.py`](../src/tdwm/training/aligned_e2e_mc_gt_lewm.py)
- tail value 实现：[`../src/tdwm/methods/goal_tail_value.py`](../src/tdwm/methods/goal_tail_value.py)
- 正式结果：[`../reports/aligned_e2e_mc_gt_lewm_cube_seed3072.md`](../reports/aligned_e2e_mc_gt_lewm_cube_seed3072.md)
- 六组 matched 归档：[`../reports/aligned_acd_cube_o50_seed3072_planning_seeds42_47.md`](../reports/aligned_acd_cube_o50_seed3072_planning_seeds42_47.md)

实现固定使用 `stable-worldmodel[all]==0.1.1`。

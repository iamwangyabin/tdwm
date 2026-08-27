# MC-GT-LeWM OGBench Cube 完整离线训练记录

训练完成：2026-08-21（Asia/Shanghai）
记录状态：单个冻结 LeWM checkpoint、单个 value-head seed 的完整离线训练；尚未接入
CEM，不构成规划性能提升结论。

## 方法

本方法命名为 **MC-GT-LeWM**（Monte Carlo Goal-Tail LeWM）。它冻结已经复现的
LeWM encoder 和 predictor，只训练一个 682,497 参数的两层 MLP：

\[
V_\theta(h_t,z_g)\rightarrow\mathbb R,
\]

其中 history 包含 3 个 LeWM latent 和前 2 个归一化 action block。训练 goal 由同一
Cube trajectory 的未来状态提供，\(\Delta\sim U(1,16)\)，监督 target 为：

\[
Y_t=(1-\gamma)\sum_{k=1}^{\Delta}\gamma^{k-1}
\frac{\lVert z_{t+k}-z_{t+\Delta}\rVert^2}{192},\qquad \gamma=0.95.
\]

训练中没有 policy、successor、TD、EMA、target network 或 LeWM joint training。
optimizer 只包含 `value.parameters()`。

## 结果

| 指标 | 未训练 value | epoch 20 / best |
| --- | ---: | ---: |
| train MSE | - | 0.00568806 |
| validation MSE | 0.14828520 | **0.00607095** |
| validation MAE | 0.30557013 | **0.05518595** |
| validation Spearman | -0.00595168 | **0.92718685** |
| validation prediction mean | 0.01727077 | 0.30885386 |
| validation target mean | 0.31259230 | 0.31259230 |

validation MSE 相对未训练 value 降低约 95.9%。最低 validation MSE 出现在 epoch 20，
因此它也是预先约定规则选出的 best checkpoint。该结果说明 scalar value 已能在真实、
冻结的 LeWM latent 上拟合并排序有限未来 MC cost；它不能说明把 value 接入 planner 后会
提升 Cube success rate。

## 完整训练协议

| 项目 | 本次值 |
| --- | --- |
| 环境与数据 | `swm/OGBCube-v0`，`quentinll/lewm-cube` |
| Stable World Model | `stable-worldmodel[all]==0.1.1` |
| 冻结 LeWM | seed 3072，epoch 10，18,034,628 参数 |
| value-head seed | 3072 |
| episode split seed | 42 |
| train / validation episodes | 9,000 / 1,000 |
| train / validation clips | 963,000 / 107,000 |
| validation pairs | 1,712,000（全部 16 个 future offsets） |
| history / frame skip | 3 / 5 |
| value hidden dimension | 512 |
| value 参数量 | 682,497 |
| batch size | 4,096 |
| epochs / optimizer steps | 20 / 4,720 |
| 每 epoch 覆盖 | 全部 963,000 个训练 clips |
| optimizer | AdamW，learning rate `3e-4`，weight decay `1e-4` |
| checkpoint selection | minimum validation MSE |
| accelerator | NVIDIA GeForce RTX 3090 |

冻结 encoder 对全部 2,010,000 个 observation 各执行一次编码，并写入
`(2010000, 192)` float32 latent cache。cache 生成耗时 940.997 秒；完成 cache 后，20 个
value 训练 epoch 耗时 176.618 秒。cache 和所有 checkpoint 均保留在外部运行目录，没有
提交到 GitHub。

## 审计信息

| Artifact | SHA-256 |
| --- | --- |
| LeWM base checkpoint | `0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579` |
| MC-GT-LeWM best checkpoint | `6ae24ffc715a761fd9a604564252d9414000d7661c175d7b6af2b1fe057d676f` |
| float32 latent cache | `9710d527f865327a7956647a84c319e5e79b6034d70fb8e1bb45716ccf05cb50` |

训练代码 Git revision 为 `ac7c397a9726fc799201559c7b60a6f4c75acffa`。正式配置是
[`configs/experiment/mc_gt_lewm_cube_train.yaml`](../configs/experiment/mc_gt_lewm_cube_train.yaml)。
运行产物保留在：

```text
/home/yabin/tdwm/outputs/mc_gt_lewm_cube_full/
```

## 下一步

下一步不是继续训练新网络，而是把 best MC-GT-LeWM value 接到原 LeWM Cube CEM：

\[
J=c(\hat z_{t+H},z_g)+V_\theta(\hat h_{t+H},z_g),
\]

固定 \(\lambda_V=1\)，保持原 checkpoint、CEM horizon、candidate 数、iteration 数、
elite 数、action block、评测 episode 和随机种子不变。完成同协议评测后，才能判断
MC-GT-LeWM 是否改善规划。只有 scalar MC tail 的规划作用得到验证后，下一项训练才是
网络结构不变的 **TD-GT-LeWM**。

# TD-GT-LeWM OGBench Cube 完整离线训练记录

训练完成：2026-08-21（Asia/Shanghai）
记录状态：单个冻结 LeWM checkpoint、单个 TD value-head seed 的完整离线训练；尚未
进行 TD-GT-LeWM CEM 评测，不构成规划性能提升结论。

## 方法

本阶段方法命名为 **TD-GT-LeWM**（Temporal-Difference Goal-Tail LeWM）。它冻结
LeWM encoder 和 predictor，保留 MC-GT-LeWM 的 682,497 参数 value 网络，并将监督
MC loss 替换为一步 TD：

\[
Y_t=(1-\gamma)c(z_{t+1},z_g)
+\gamma\mathbf{1}[z_{t+1}\ne z_g]V_{\bar\theta}(h_{t+1},z_g),
\qquad \gamma=0.95.
\]

训练 goal 仍从同一轨迹按 \(\Delta\sim U(1,16)\) 采样。在线 value 随机初始化，target
network 从在线网络复制并以 `0.995` EMA 更新；没有从 MC-GT-LeWM warm-start。LeWM
完全冻结，optimizer 只包含在线 `value.parameters()`，到达 hindsight goal 时停止
bootstrap。有限未来 MC return 只作为固定 validation metric，不参与优化。

## 结果

最低 validation MC MSE 出现在 epoch 19，因此按预先锁定规则选择 epoch 19 checkpoint：

| 指标 | 未训练 value | epoch 19 / best | epoch 20 |
| --- | ---: | ---: | ---: |
| validation MC MSE | 0.20912155 | **0.00864504** | 0.00865675 |
| validation MC MAE | 0.38449873 | **0.07203945** | 0.07292373 |
| validation MC Spearman | -0.04334504 | 0.90422519 | **0.90725171** |
| validation prediction mean | -0.07126415 | 0.33153208 | 0.33660316 |
| validation MC target mean | 0.31259231 | 0.31259231 | 0.31259231 |
| validation TD MSE | 0.00667325 | 0.00272754 | 0.00268761 |

TD value 明确学到了长期 cost：固定 validation MC MSE 相对初始化下降约 95.9%，排序
相关性达到 0.90 以上。但在相同数据、网络和 split 上，它仍弱于 MC-GT-LeWM 的
validation MC MSE `0.00607095`、MAE `0.05518595` 和 Spearman `0.92718685`。因此目前
证据支持“TD 训练链路可用”，不支持“TD 比有限未来 MC 监督更好”。

## 完整训练协议

| 项目 | 本次值 |
| --- | --- |
| 环境与数据 | `swm/OGBCube-v0`，`quentinll/lewm-cube` |
| Stable World Model | `stable-worldmodel[all]==0.1.1` |
| 冻结 LeWM | seed 3072，epoch 10，18,034,628 参数 |
| TD value-head seed | 3072 |
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
| target EMA decay | 0.995 |
| optimizer | AdamW，learning rate `3e-4`，weight decay `1e-4` |
| checkpoint selection | minimum validation MC MSE |
| head 训练耗时 | 177.136 秒 |
| accelerator | NVIDIA GeForce RTX 3090 |

训练复用 MC-GT-LeWM 已生成并审计的 `(2010000, 192)` float32 latent cache。正式
checkpoint 保存在线 value、target value、optimizer、DataLoader shuffle generator 和
goal-offset generator 状态。另一次两阶段 smoke 已验证从 epoch checkpoint 恢复后可以
继续训练。

## 审计信息

| Artifact | SHA-256 |
| --- | --- |
| LeWM base checkpoint | `0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579` |
| TD-GT-LeWM best checkpoint | `cb2b88014a969c37f791fa6dbffdfcd168869587eeb32a909412f75b47c43c18` |
| TD-GT-LeWM last checkpoint | `175ade3d5f04356899107f678dcc2038d034fa2b98a5b403b5d0b0ad83cdbc5c` |
| float32 latent cache | `9710d527f865327a7956647a84c319e5e79b6034d70fb8e1bb45716ccf05cb50` |
| episode split | `6a0d27343022ac2cb844dbc637127fdb95e89d0462ab4a7ddbeb3b52e6e0487d` |

训练代码 Git revision 为 `8710baaba845eb89b1e968670388436c311cd373`。正式配置是
[`configs/experiment/td_gt_lewm_cube_train.yaml`](../configs/experiment/td_gt_lewm_cube_train.yaml)。
可恢复的正式运行产物保留在：

```text
/home/yabin/tdwm/outputs/td_gt_lewm_cube_full_resumable/
```

## 当前判断

目前整条研究链路给出的结论是：

1. MC-GT-LeWM 在真实冻结 latent 上学习良好，但同 checkpoint 的 50-episode CEM
   评测仅从 LeWM 的 72% 变为 74%，配对净增 1 个任务，证据不足。
2. TD-GT-LeWM 成功传播了长期 cost，但固定 MC validation 指标弱于 MC-GT-LeWM。
3. 尚未运行 TD-GT-LeWM 的同协议 CEM，因此不能判断 TD value 对控制是否有不同于
   offline metric 的收益。

所以该方法目前是**可运行、可复现、值得继续诊断的研究原型**，但还没有达到“优于
LeWM baseline”的项目里程碑。下一项受控实验应把 epoch 19 TD value 接入完全相同的
CEM 和 50 个 start--goal pair，除此之外不改变任何变量。

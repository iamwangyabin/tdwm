# LeWM OGB Cube：seed 3072 单次复现实验记录

评测完成：2026-08-17（Asia/Shanghai）  
记录状态：单训练 seed 的初步复现，**不是**已完成的 baseline 结论。

## 结果

在 50 个固定的 OGB Cube start--goal 对上，epoch 10 checkpoint 的成功率为
**54.0%（27 / 50）**；整次 CEM 评测耗时 **99.32 s**。本次没有生成视频。

这看起来低于公开 LeWM 论文 Figure 6 中 Cube 的约 **74%**（相差约 **20 个百分
点**）。但这只是与论文图读值的**间接对照**，不能据此写成“本项目复现的 LeWM
显著更差”：目前只有一个训练 seed，而且训练数据读取和图像存储与论文/发布训练的
精确运行细节尚未逐项对齐。

## 训练产物与评测协议

| 项目 | 本次值 |
| --- | --- |
| 方法 | LeWM |
| 环境 | `swm/OGBCube-v0` |
| 训练 seed | 3072 |
| checkpoint | epoch 10，`global_step=127960` |
| 参数量 | 18,034,628 |
| 数据 | `quentinll/lewm-cube`，10,000 episodes、2,010,000 transitions |
| 图像数据格式 | Lance，JPEG quality 100 |
| 训练读取 | 顺序读取 Lance block、block 内打乱、2 workers、prefetch 2；关闭 `torch.compile` |
| 评测 goal offset | 25（同一 expert trajectory） |
| 评测 episode 数 | 50，planning seed 42，无放回抽样 |
| CEM | horizon 5、300 candidates、10 iterations、30 elites、action block 5 |
| 环境执行 | frame skip 5；每次规划后执行 25 个环境 step；episode budget 50 |
| 渲染后端 | EGL（`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`） |

评测配置已版本控制在
[`configs/experiment/lewm_cube_seed3072_o25.yaml`](../configs/experiment/lewm_cube_seed3072_o25.yaml)。
评测使用的 checkpoint 权重 SHA-256 是
`7fddc5117e142e70a9bf93170c6db10089ab0f3582313359040d8da5f0a2bdad`。
评测协议配置的 Git revision 是 `de1c690`。

## 训练与验证末值

| 指标 | 值 |
| --- | ---: |
| train/loss | 0.13555546 |
| train/prediction_loss | 0.01771848 |
| train/sigreg_loss | 1.30928361 |
| validation/loss | 0.22475678 |
| validation/prediction_loss | 0.01425450 |
| validation/sigreg_loss | 2.33890247 |

## 如何解读与下一步对照

本次结果首先说明：这个 checkpoint 在锁定的 50-goal CEM 协议下能稳定完成 27 个
Cube 任务，训练和 EGL 评测链路均成功结束。它**不**能说明 LeWM 本身退化，也不能
成为提出方法的比较基线。

与公开约 74% 数字仍有下列不能忽略的差异：

- 这里只有训练 seed 3072；必须至少补齐预先锁定的 seed 0、42、3072，并报告均值和离散度。
- 本次把原始数据转为 JPEG-100 Lance，并采用 block-locality 读取/块内打乱来避免云盘随机
  I/O 停顿。数值字段保持一致，但读取次序和有损像素编码仍是训练实现差异。
- 此处的 74% 是论文图中的公开结果，不是同一 checkpoint、同一数据存储、同一训练随机性
  下重新跑出的受控对照。后续需要用同一评测器复跑公开 checkpoint，或严格复现其训练入口，
  才能判断这 20 个点来自何处。

云端原始轻量结果位于 run ID
`seed_3072_blockshuffle_w2_pf2_valblock_resume_e01_evaluation_o25_egl` 的 `results.json`；
模型、数据和完整日志按项目规则留在受控外部存储，未提交到 Git。

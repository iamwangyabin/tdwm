# LeWM PushT baseline 训练记录（进行中）

首次启动：2026-08-10 21:42（Asia/Shanghai）<br>
本次更新：2026-08-11 00:27（Asia/Shanghai）

## 结论摘要

LeWM PushT 的单 seed、10 epoch 训练已经在 Gemini 开发环境启动。第 1 个 epoch
的训练部分已完成，过程未观察到 NaN、OOM、异常退出或 checkpoint 写入失败；完整
验证仍在运行。因此，本记录只证明训练链路能够持续运行，**不代表 baseline 已经
复现完成，也不包含成功率或长时 rollout 结论**。

这次运行启动于当前包使用规则确定之前，使用了上游源码快照和兼容 adapter。它不
满足当前“只基于已安装的 `stable-worldmodel[all]==0.1.1` 和公开 API 开发”的验收
要求。结果只能作为工程诊断和后续合规复现的参照，不能作为最终论文 baseline。

## 软件与硬件

- Gemini 开发环境，单张 `B1.gpu.large`，可见显存 24,258 MiB；
- Python 3.11.8；
- PyTorch 2.2.2+cu121；
- `stable-worldmodel==0.1.1`；
- 过渡运行使用的上游源码快照：`67017b79ef194e96fd96d201fa3ba51ffff62775`；
- 随机种子：3072；
- 混合精度：BF16；
- 模型参数量：18,034,478。

## 数据与划分

本仓库和 GitHub **不保存任何数据集文件**。以下只记录复现所需的非数据元信息：

- 数据标识：LeWM PushT expert train；
- 原始 HDF5 SHA-256：
  `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`；
- 解析得到 18,685 episodes、2,336,736 transitions；
- 长度为 20 的候选 clips：1,981,721；
- 固定划分：90% train、10% validation；
- train clips：1,783,548，batch size 128，每个 epoch 13,933 optimizer steps；
- validation batches：1,549；
- 图像尺寸：224×224；
- frame skip：5。

为规避云盘随机读取停顿，本次运行把原始 HDF5 转换为 Stable World Model 支持的
Lance 格式。action、proprioception、state 和 episode 长度抽样核对一致；图像使用
JPEG quality 95，抽样 PSNR 约 52.9–53.5 dB。该有损图像转换是相对原始 HDF5 的
实验偏差，正式公平对比必须对所有方法统一使用同一转换产物和预处理。

## 训练配置

| 项目 | 值 |
| --- | ---: |
| epochs | 10 |
| batch size | 128 |
| train workers | 6 |
| prefetch factor | 2 |
| optimizer | AdamW |
| learning rate | 5e-5 |
| weight decay | 1e-3 |
| gradient clipping | 1.0 |
| history frames | 3 |
| prediction frames | 1 |
| encoder | ViT-Tiny, from scratch |
| predictor depth | 6 |
| predictor dropout | 0.1 |
| SIGReg weight | 0.1 |
| checkpoint interval | 1,000 optimizer steps |

训练和验证使用同一个固定数据划分。为适应容器 32 GiB 内存上限，validation loader
使用单进程读取；这不改变验证样本，但显著增加了验证耗时。

## 截止本次更新的状态

- Epoch 0 训练：13,933 / 13,933 steps，约 1 小时 44 分；
- 稳态训练吞吐：约 2.2–2.4 steps/s；
- 训练期 GPU 利用率：通常约 98%；
- GPU 显存：约 13.49 GiB；
- 最近训练总损失：0.16631；
- 最近 prediction loss：0.03154；
- 最近 SIGReg loss：1.34375；
- Epoch 0 验证：约 831 / 1,549 batches，仍在进行；
- 最近完整写入的 checkpoint：step 13,000，约 206.8 MiB；
- 未观察到 NaN、OOM、Traceback、跳批或 checkpoint 原子写入失败。

上述 loss 是训练过程中的最近观测，不是最终 epoch 指标。验证完成前不得用当前
逐 batch 数值与论文结果比较。

## 已解决的工程问题

1. 持久云盘上的 HDF5 随机读取会出现长时间停顿，改为 Lance 后训练吞吐稳定。
2. CUDA 初始化后 fork 数据进程会触发 worker 异常，过渡 adapter 改用 forkserver。
3. 32 workers 超过容器真实 32 GiB cgroup 上限；最终使用 6 workers。
4. 同时保留 train/validation worker pool 会导致内存不足；validation 改为主进程读取。
5. 预取系数 3 使内存余量不足；改为 2 后完成了多次 checkpoint 保存。

这些处理属于本次过渡实验的运行适配，不应被复制成项目内的上游框架分叉。后续
必须通过已安装包的公开 API 和项目自身轻量 adapter 重新实现合规入口。

## 尚未完成

- 其余 9 个 epochs；
- 最终 held-out validation 汇总；
- checkpoint 恢复后一致性测试；
- one-step prediction 的正式指标；
- 长时 open-loop rollout；
- 使用统一 CEM 预算的 PushT success rate；
- 多随机种子均值、标准差和最差 seed；
- 与 PLDM、DINO-WM 等 baseline 的同协议比较。

在以上项目完成前，不得写成“LeWM baseline 已复现”，也不得据此声称 TDWM 优于
任何 baseline。

## Artifact 策略

数据集、Lance 文件、模型、checkpoint、完整日志和视频只保存在受控的云端存储，
不进入 GitHub。本仓库只保存本报告、实验配置和最终轻量汇总指标。远端 artifact
路径会随平台任务变化，因此不作为代码默认值；最终报告通过 run ID、代码版本和
校验信息关联外部 artifact。

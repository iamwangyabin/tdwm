# 四环境复现配置

配置分为环境、方法和锁定的实验协议：

```text
configs/
├── envs/            # 每个环境的环境、数据和评测配置
├── methods/         # 每个方法的模型和训练配置
└── experiment/      # 已锁定、可直接执行和审计的实验协议
```

训练时直接选择一个环境配置和一个方法配置进行组合。进入实际运行的单元必须在
`experiment/` 中锁定数据、checkpoint、episode 选择、规划预算和成功定义；这类文件
是运行协议，不是重复维护的实验矩阵。
完整实验仍然是 4 个环境 × 7 个方法；LeWM Cube 复现锁定训练种子
`0, 42, 3072`。

## 当前矩阵状态

| 方法 | PushT | Cube | Reacher | Two-Room |
| --- | --- | --- | --- | --- |
| LeWM | ready | ready | ready | ready |
| PLDM | ready | ready | ready | ready |
| DINO-WM（pixels-only） | ready | ready | ready | ready |
| GCBC | ready | ready | ready-new | ready |
| GCIVL | ready | ready | ready-new | ready |
| GCIQL | ready | ready | ready-new | ready |
| TD-MPC2 | protocol-gated | protocol-gated | protocol-gated | protocol-gated |

`ready-new` 表示训练配置完整，但论文没有发布该方法在 Reacher 上的 checkpoint
或数值，因此这是新的受控复现，不是已有论文结果的重复运行。

TD-MPC2 被显式门控，是因为它需要 reward/Q 学习，而 LeWM 四环境主协议是离线、
reward-free、任意目标条件控制。没有先确定 goal-conditioned reward relabeling 或独立
在线训练协议，直接启动 TD-MPC2 会产生不可比较的结果。其四个运行单元仍保留在
研究范围中，但训练入口必须读取 `methods/tdmpc2.yaml` 的协议门控并拒绝默认启动。

## 采用的协议

- 固定依赖：`stable-worldmodel[all]==0.1.1`；
- 主结果使用相同离线数据、训练/验证划分、评测 episode、起点、目标偏移和预算；
- LeWM Cube 训练随机种子：`0, 42, 3072`；
- 原始 LeWM 训练按 sequence clip 使用每个训练 seed 做 90/10 随机划分，并保存
  每次运行的确切索引；后续方法比较必须复用相应 seed 的索引；
- 每次评测 50 条轨迹，目标偏移 25 个环境步，执行预算 50 个环境步；
- world model 统一使用 CEM：300 candidates、30 elites、horizon 5、action block 5；
- PushT 使用 30 次 CEM 迭代，其他环境使用 10 次；
- world model 训练 10 epochs，符合 LeWM 论文附录中的四环境协议；
- policy baseline 使用其发布配置的训练预算：GCBC/GCIVL/GCIQL 为 100 epochs；
- 主要 DINO-WM 结果不使用 proprioception；`DINO-WM+prop` 应作为单独消融，不能
  覆盖 pixels-only 主结果。

## 论文与发布配置的差异

LeWM 论文明确写明四个 world model 均训练 10 epochs；公开仓库的通用训练 YAML
当前默认 100 epochs。本套件使用论文值 10。LeWM 的 SIGReg 权重采用发布配置中的
`0.09`；论文正文将默认值表述为约 `0.1`。这两个来源都记录在 method 配置的
`provenance` 字段中。

## 数据放置

官方 Hugging Face 数据是 `.zst` 压缩归档，`stable-worldmodel==0.1.1` 不会直接
读取这些压缩包。先解压，再把得到的 HDF5 文件放到
`$STABLEWM_HOME/datasets/` 下，保持对应 `envs/*.yaml` 中记录的相对名称。

当前仓库已有 PushT 压缩包：

```text
data/lewm-pusht/pusht_expert_train.h5.zst
```

不要把数据集、checkpoint 或训练输出提交到 Git。其余三个官方压缩包合计很大，
正式下载和解压前应先确认 GPU 机器的磁盘空间。

## 执行边界

环境、方法配置与 `experiment/` 下的锁定协议是实验事实来源。当前已实现官方 LeWM
checkpoint 的 Cube O25 评测入口：

```bash
export STABLEWM_HOME=/path/to/persistent/stable_worldmodel
export TDWM_CUBE_DATASET=/path/to/ogbench/cube_single_expert.h5
python -m pip install --no-deps -e .
python scripts/evaluate.py \
  --config configs/experiment/lewm_cube_checkpoint_o25.yaml \
  --output-dir outputs/lewm_cube_official_checkpoint_o25
```

正式评测前使用相同输出目录运行一次 `--smoke`。冒烟运行只执行 1 条 episode、8 个
候选和 1 次 CEM 迭代，并缓存从完整数据计算的 action normalization；随后的正式运行
会复用该统计量并以锁定协议覆盖冒烟清单和结果：

```bash
python scripts/evaluate.py --smoke \
  --config configs/experiment/lewm_cube_checkpoint_o25.yaml \
  --output-dir outputs/lewm_cube_official_checkpoint_o25
```

入口通过 `import stable_worldmodel as swm` 使用 `0.1.1` 的公开 API，不复制上游
baseline 源码。每次运行会先写入协议清单和确切的 episode/start/goal 索引，再开始
规划；逐 episode success 和完整聚合结果写入 `results.json`。

官方 checkpoint 评测通过后，使用锁定的原始训练配置进行多 seed 重训。正式训练
前必须先运行冒烟训练，再用 `--resume required` 验证 Lightning checkpoint 可以恢复：

```bash
python scripts/train.py --smoke --resume never --seed 0 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET"
python scripts/train.py --smoke --resume required --seed 0 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET"
python scripts/train.py --seed 0 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET"
```

训练运行会保存协议与运行时清单、确切 split 索引、每 epoch 的 Lightning checkpoint
和 Stable World Model 可加载的导出权重。数据、checkpoint 和原始日志只保存在运行
目录，不进入 Git。

Cube 原始 HDF5 的 pixels 每 100 帧压缩为一个 chunk，在远程挂载上进行随机 clip
训练会产生严重读取放大。可以先使用无损重排入口把 pixels 改为每帧一个 chunk；
它保留全部列、episode 边界、dtype 和像素值：

```bash
python scripts/rechunk_cube_hdf5.py \
  data/lewm-cube/cube_single_expert.h5 \
  data/lewm-cube/cube_single_expert_chunk1.h5
```

转换完成后会逐列抽样校验并生成相邻的 manifest。优化布局的确切大小和 SHA-256
锁定在 `experiment/lewm_cube_train.yaml`。该布局适合逐像素严格复现，但在云端远程
挂载上仍可能受 HDF5 随机读取延迟限制。

正式多 seed 快速训练可以再通过 `stable-worldmodel==0.1.1` 的公开转换 API 生成
Lance 数据集：

```bash
python scripts/convert_cube_lance.py \
  data/lewm-cube/cube_single_expert_chunk1.h5 \
  data/lewm-cube/cube_single_expert_jpeg100.lance
```

转换固定使用 JPEG 质量 100，并生成相邻的 `.lance.manifest.json`。训练入口同时接受
锁定大小的 HDF5 文件和带有效 manifest 的 `.lance` 目录；缺少 manifest、转换版本
不符、图像质量不是 100、`action` 不是精确值，或 `observation` 不是源数据的确定性
float32 转换时，Lance 数据会被拒绝。JPEG-100 保持图像分辨率，但不是逐像素无损，
因此必须把这类运行标记为快速数据变体，并保证所有对比 seed 使用同一个转换结果。

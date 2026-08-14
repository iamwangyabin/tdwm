# TDWM

TDWM 基于固定版本的 `stable-worldmodel[all]==0.1.1` 开展 world model
基线复现和后续方法研究。锁定的实验协议、数据来源与评测参数见
[`configs/README.md`](configs/README.md)。

## 趋动云上的 LeWM Cube 快速训练

Cube 的训练样本是随机 sequence clip。即使把原始 HDF5 无损重排为单帧 pixel
chunk，直接从 `/gemini/data-*` 的远程挂载随机读取仍会让 GPU 等待 I/O。正确做法
不是为每个 seed 复制数据，而是在一个训练任务开始时把优化后的 HDF5 顺序读入
`/dev/shm` 一次，随后在同一任务内完成冒烟、恢复检查和全部训练 seed。

### 前提

- 把 `cube_single_expert_chunk1.h5` 作为持久数据集挂载。生成方式和无损校验见
  [`configs/README.md`](configs/README.md)。
- 使用离线训练而不是开发环境进行正式训练，并挂载代码、Cube 数据集和结果集。
- 选择 `/dev/shm` 可用空间至少为 `90 GiB`、并能为训练进程额外保留足够内存的
  资源规格。数据文件本身为 `74,104,077,358` bytes。
- 所有 checkpoint、日志和结果写入 `$GEMINI_DATA_OUT`。不要写回只读的数据集
  挂载，也不要把唯一结果留在 `/dev/shm` 或其他临时目录。

### 单任务启动命令

下面的命令只在 RAM 中不存在已校验副本时进行一次顺序复制。三个 seed 共用同一
副本；任务因故重新调度时，数据会重新预热，但 `--resume auto` 会从持久结果目录
恢复正式训练。

```bash
set -euo pipefail

: "${GEMINI_CODE:?GEMINI_CODE is required}"
: "${GEMINI_DATA_IN1:?GEMINI_DATA_IN1 is required}"
: "${GEMINI_DATA_OUT:?GEMINI_DATA_OUT is required}"
test -d "$GEMINI_CODE/tdwm"
test -r "$GEMINI_DATA_IN1/cube_single_expert_chunk1.h5"
test -d "$GEMINI_DATA_OUT" && test -w "$GEMINI_DATA_OUT"

repo="$GEMINI_CODE/tdwm"
source_h5="$GEMINI_DATA_IN1/cube_single_expert_chunk1.h5"
staged_h5=/dev/shm/cube_single_expert_chunk1.h5
partial_h5="${staged_h5}.partial"
expected_size=74104077358
expected_sha=3cf6477768f1a2979acefa3aeb6c27c45422b8b6fbce8527419943d3e679a245
headroom=$((16 * 1024 * 1024 * 1024))

if [[ ! -f "$staged_h5" ]]; then
  available_shm=$(df --output=avail -B1 /dev/shm | tail -n 1)
  if (( available_shm < expected_size + headroom )); then
    echo "Need at least $((expected_size + headroom)) bytes free in /dev/shm" >&2
    exit 1
  fi
  rm -f "$partial_h5"
  dd if="$source_h5" of="$partial_h5" bs=16M status=progress
  test "$(stat -c %s "$partial_h5")" -eq "$expected_size"
  test "$(sha256sum "$partial_h5" | cut -d ' ' -f 1)" = "$expected_sha"
  mv "$partial_h5" "$staged_h5"
fi

test "$(stat -c %s "$staged_h5")" -eq "$expected_size"
test "$(sha256sum "$staged_h5" | cut -d ' ' -f 1)" = "$expected_sha"

export TDWM_CUBE_DATASET="$staged_h5"
export TDWM_RUN_ROOT="$GEMINI_DATA_OUT/tdwm"
export STABLEWM_HOME="$GEMINI_DATA_OUT/stable_worldmodel"
run_dir="$TDWM_RUN_ROOT/lewm_cube_training"
mkdir -p "$run_dir/logs" "$STABLEWM_HOME"

cd "$repo"
python -c "import stable_worldmodel as swm; print(swm.__file__)"

python scripts/train.py --smoke --resume never --seed 3072 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/logs/smoke_seed_3072.log"

python scripts/train.py --smoke --resume required --seed 3072 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir" \
  2>&1 | tee -a "$run_dir/logs/smoke_seed_3072.log"

for seed in 0 42 3072; do
  python scripts/train.py --resume auto --seed "$seed" \
    --config configs/experiment/lewm_cube_train.yaml \
    --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir" \
    2>&1 | tee "$run_dir/logs/seed_${seed}.log"
done
```

### 为什么采用这个布局

- `chunk1 HDF5` 保留原始像素值，并消除原始 100 帧 chunk 对随机 clip 的解压读取
  放大；它解决的是 HDF5 布局问题。
- RAM 预热解决远程挂载的随机请求延迟。当前趋动云实例的顺序读取实测约为
  `57-87 MB/s`，所以一次预热约需 `14-22` 分钟，不能把这段时间重复到每个 seed。
- 不使用 Lance 作为严格复现数据，因为 `stable-worldmodel==0.1.1` 的公开
  `LanceWriter` 会把图像列重新编码为 JPEG。
- 不把数据塞进训练镜像；超大镜像只会把数据传输成本转移到镜像拉取和启动阶段。

`/dev/shm` 会随实例消失。不要在预热或训练期间停止开发环境、删除离线任务或重启
实例；发生重调度后，应重新预热 RAM，并从 `$GEMINI_DATA_OUT` 中的 checkpoint
恢复。

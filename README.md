# TDWM

TDWM runs auditable world-model experiments on the pinned
`stable-worldmodel[all]==0.1.1` platform. PushT currently has executable,
single-seed adapters for LeWM, PLDM, DINO-WM, GCBC, GCIVL, and GCIQL. TD-MPC2
remains protocol-gated because its online reward/Q objective is not comparable
to this offline experiment group.

The dataset and all generated checkpoints remain outside Git. With the PushT
Lance dataset already available, launch the configured run with:

```bash
python scripts/train.py \
  --env pusht \
  --method lewm \
  --seed 3072 \
  --dataset /path/to/pusht_expert_train.lance \
  --run-root /path/to/persistent/tdwm-runs
```

The command reads `configs/envs/pusht.yaml` and the selected method YAML as the
experiment's source of truth. It refuses to run if the imported
`stable_worldmodel` is not the installed `0.1.1` distribution. To keep four
GPUs occupied while the five additional methods train, use the resumable queue:

```bash
python scripts/launch_pusht_parallel.py \
  --seed 3072 \
  --dataset /path/to/pusht_expert_train.lance \
  --run-root /path/to/persistent/tdwm-runs
```

The launcher starts at most one method per GPU, writes its state and per-method
logs under `RUN_ROOT/launcher/pusht_parallel_seed3072/`, and starts queued
methods whenever a GPU becomes free. GCIVL and GCIQL preserve their released
two-stage value/critic-then-policy training protocols.

If the training node cannot reach Hugging Face, place the official
`facebook/dinov2-small` snapshot in persistent storage and set
`TDWM_DINO_BACKBONE` to that directory before launching. The path is injected
at runtime and is recorded in experiment metadata; it is never hard-coded in
the repository.

## Reproducible GPU container

The repository Dockerfile builds on the CUDA-enabled image already supplied by
the training platform. It does not select, install, or upgrade PyTorch,
torchvision, or CUDA. The build records their versions before installing TDWM
and fails if dependency resolution changes any of them.

The added layer installs the required `stable-worldmodel[all]==0.1.1`
distribution and pins its critical non-accelerator dependencies in
`docker/constraints.txt`. The supplied base image must already contain
Python 3.10 or newer, CUDA-enabled PyTorch, and its matching torchvision.

Build the TDWM layer on the existing GPU image:

```bash
docker build \
  --build-arg BASE_IMAGE=your-existing-gpu-image:tag \
  -t tdwm:runtime .
```

Verify that the host exposes an NVIDIA GPU to the container:

```bash
docker run --rm --gpus all tdwm:runtime \
  python -c "import torch, stable_worldmodel as swm; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), swm.__file__)"
```

Run a single PushT method with the dataset mounted read-only and checkpoints
mounted to persistent storage:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -v /local/pusht_expert_train.lance:/datasets/pusht_expert_train.lance:ro \
  -v /local/tdwm-runs:/workspace/runs \
  -v /local/stable-worldmodel-cache:/workspace/cache/stable_worldmodel \
  tdwm:runtime \
  python scripts/train.py \
    --env pusht \
    --method lewm \
    --seed 3072 \
    --dataset /datasets/pusht_expert_train.lance \
    --run-root /workspace/runs
```

Dataset and run mounts are deliberately external to the image. Rebuilding the
container therefore cannot delete downloaded datasets or checkpoints.

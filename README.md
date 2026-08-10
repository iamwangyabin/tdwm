# TDWM

TDWM runs auditable world-model experiments on the pinned
`stable-worldmodel[all]==0.1.1` platform. The first executable experiment is a
single-seed LeWM baseline on PushT.

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

The command reads `configs/envs/pusht.yaml` and `configs/methods/lewm.yaml` as
the experiment's source of truth. It refuses to run if the imported
`stable_worldmodel` is not the installed `0.1.1` distribution.

# LS-LeWM: Local Dynamics with a Policy-Conditioned Successor Tail

## Status

LS-LeWM is a proposed method implemented independently of the upstream LeWM
baseline. It reuses the public `stable_worldmodel==0.1.1` LeWM class, data API,
CEM solver, policy wrapper, checkpoint API, and evaluation path. It does not
modify the installed package or the repository's LeWM baseline implementation.

The method is logically closed, but improved control performance is an
experimental hypothesis, not a theorem or a current result.

## State Used by the Method

LeWM's predictor is history dependent, so the Bellman state is not a single
latent vector. For history length `m`, define

\[
h_t=(z_{t-m+1:t},a_{t-m+1:t-1}), \qquad z_t=\phi(o_t).
\]

The local LeWM model predicts the next latent from this context and an action:

\[
\hat z_{t+1}=F(h_t,a_t).
\]

An executable goal-conditioned continuation policy is learned by hindsight
behavior cloning:

\[
\pi_\eta(h_t,z_g)\approx a_t,
\]

where every later state in the same offline clip can provide `z_g`. This policy
keeps the successor target close to the demonstrated action support and makes
the policy used in the Bellman equation available at inference time.

## Successor Object

Raw first moments of `z` cannot recover squared latent goal distance. LS-LeWM
therefore uses the exact feature lift

\[
\chi(z)=\left[\frac{z}{\sqrt d},\frac{\lVert z\rVert^2}{d},1\right]
\]

and goal weights

\[
w(z_g)=\left[-\frac{2z_g}{\sqrt d},1,
\frac{\lVert z_g\rVert^2}{d}\right].
\]

They satisfy

\[
w(z_g)^\top\chi(z)=\frac{\lVert z-z_g\rVert^2}{d}.
\]

The learned vector successor is

\[
G^{\pi_g}(h_t,a_t,z_g)
=(1-\gamma)\mathbb E\left[
\sum_{k=0}^{\tau_g-t-1}\gamma^k\chi(z_{t+k+1})
\mid h_t,a_t,\pi_g
\right].
\]

Its one-step Bellman equation is

\[
G(h_t,a_t,z_g)=(1-\gamma)\chi(z_{t+1})
+\gamma(1-d_{t+1})G_{\rm ema}
(h_{t+1},\pi_{\rm ema}(h_{t+1},z_g),z_g),
\]

where `d_{t+1}=1` when the next state is the hindsight goal. At a terminated
goal context, `G(h_g, pi(h_g,z_g), z_g)=0`. Uniform weighting over goal offsets
prevents the fewer long-offset pairs from being overwhelmed by short offsets.

## Training Objective

The shared encoder, LeWM predictor, successor head, and continuation policy are
trained jointly with

\[
\mathcal L=
\mathcal L_{\rm local}
+\beta\mathcal L_{\rm SIGReg}
+\lambda_G(\mathcal L_{\rm TD}+\lambda_b\mathcal L_{\rm terminal})
+\lambda_\pi\mathcal L_{\rm BC}.
\]

`SIGReg` is the original LeWM anti-collapse mechanism and remains the only
anti-collapse regularizer. The TD target is stop-gradient and uses EMA copies
of both the successor and continuation policy. BC inputs are detached from the
encoder, while the successor loss is allowed to shape the shared latent. A
small auxiliary batch applies the same TD equation to teacher-forced one-step
LeWM contexts; this reduces the real-latent versus imagined-latent mismatch
without introducing a different target.

## MPC Inference

For a CEM candidate prefix `a_{t:t+H-1}`, LeWM predicts
`z_hat_{t+1:t+H}`. LS-LeWM scores it with

\[
J_H=(1-\gamma)\sum_{k=1}^{H}\gamma^{k-1}
\frac{\lVert\hat z_{t+k}-z_g\rVert^2}{d}
+\gamma^H w(z_g)^\top
G(\hat h_{t+H},\pi(\hat h_{t+H},z_g),z_g).
\]

The prefix and tail have no overlapping state. The same continuation policy
rolls out any missing CEM warm-start actions. MPC still matters: `G` evaluates
one proposed first action followed by a fixed continuation policy, while CEM
can optimize an arbitrary corrective prefix and replan after every action
block. Using the actor alone is possible but is not the LS-LeWM inference rule.

## Why the Design Is Feasible

- The Bellman object, its continuation policy, and the inference tail refer to
  the same executable closed-loop process.
- The lifted vector successor gives an exact linear readout for the latent MSE
  used by MPC; summing raw latent vectors would not.
- Normalization by `1-gamma` keeps prefix and tail scales compatible.
- Hindsight GCBC limits the main offline extrapolation risk of evaluating an
  unsupported continuation policy.
- LeWM's local arbitrary-action model is retained for the explicit MPC prefix;
  the successor is a terminal heuristic rather than a replacement world model.

The remaining risks are empirical: finite offline action coverage, actor error,
bootstrapping error, model-rollout distribution shift, and the possibility that
a low-capacity successor does not exert enough useful pressure on the encoder.
These risks require controlled runs and cannot be resolved by the equations
alone.

## Implementation Map

- Core heads and Bellman target: `src/tdwm/methods/local_successor.py`
- CEM cost and actor warm start: `src/tdwm/adapters/local_successor.py`
- Training and paired exports: `src/tdwm/training/local_successor.py`
- Evaluation: `src/tdwm/evaluation/local_successor.py`
- Locked protocols: `configs/experiment/ls_lewm_cube_*.yaml`
- Entry points: `scripts/train_ls_lewm.py` and `scripts/evaluate_ls_lewm.py`

Run a training smoke test before a full run:

```bash
python scripts/train_ls_lewm.py --smoke --resume never --seed 0 \
  --dataset "$TDWM_CUBE_DATASET"
python scripts/train_ls_lewm.py --seed 0 --resume auto \
  --dataset "$TDWM_CUBE_DATASET"
```

Evaluate a paired base-model export and successor-head export through CEM/MPC:

```bash
python scripts/evaluate_ls_lewm.py --smoke \
  --dataset "$TDWM_CUBE_DATASET" \
  --base-checkpoint-path /path/to/exports/checkpoints/epoch_10 \
  --heads-checkpoint-path /path/to/ls_lewm/epoch_10.pt
```

The evaluator verifies the base checkpoint SHA-256 stored in the heads export,
so heads cannot silently be paired with a LeWM checkpoint from another run.

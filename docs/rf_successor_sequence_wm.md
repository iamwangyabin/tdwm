# Reward-Free Successor Sequence WM

This is the S-only ablation of RF-Successor-LeWM. It uses the public
`stable-worldmodel==0.1.1` LeWM container for visual encoding and the shared
dataset, checkpoint, CEM, and evaluation infrastructure. It does not train or
invoke the LeWM autoregressive dynamics predictor.

## Learning object

For latent history `H_t` and an externally supplied action prefix `A_1:h`, a
causal GRU predicts one lifted future moment per horizon:

\[
\hat m_h=G_\theta(H_t,A_{1:h}),\qquad
m_h^\star=\phi(z_{t+h}),
\]

where

\[
\phi(z)=\left[z/\sqrt d,\ \lVert z\rVert_2^2/d,\ 1\right].
\]

The model output is the normalized discounted cumulative sequence

\[
\hat S_h=\frac{1}{Z_h}\sum_{j=1}^{h}\gamma^{j-1}\hat m_j,
\qquad Z_h=\sum_{j=1}^{h}\gamma^{j-1}.
\]

This construction makes recurrence consistency exact rather than learned.
Every future moment can be recovered from adjacent outputs:

\[
\hat m_h=\frac{Z_h\hat S_h-Z_{h-1}\hat S_{h-1}}{\gamma^{h-1}}.
\]

The first `d` coordinates recover the corresponding future latent. Thus
`h=1` supplies the one-step target and all horizons jointly supply the
multi-step targets without separate latent losses.

## Objective

The complete objective has one predictive term and one anti-collapse term:

\[
\mathcal L=
\frac{1}{K}\sum_{h=1}^{K}
d_{\rm balanced}(\hat S_h,S_h^\star)
+\lambda_{\rm SIGReg}\operatorname{SIGReg}(Z).
\]

Targets come from the online encoder. There is no EMA teacher, local LeWM
loss, autoregressive latent loss, recurrence loss, reward, goal, policy, or TD
bootstrap.

## Planning

For every CEM candidate, the head directly predicts `S_K`. A newly encoded
goal is queried through the exact squared-distance linear functional

\[
C(A,g)=\hat S_K(A)^\top w(g).
\]

The primary configuration has terminal latent weight zero, so planning does
not execute an autoregressive LeWM rollout.

## Commands

```bash
python scripts/train_rf_successor_sequence_wm.py --smoke --resume never \
  --seed 0 --dataset /path/to/cube.lance

python scripts/evaluate_rf_successor_sequence_wm.py --smoke \
  --dataset /path/to/cube.h5 \
  --base-checkpoint-path /path/to/base/export \
  --successor-checkpoint-path /path/to/rf_successor_sequence_wm/epoch_10.pt
```

No performance claim is made until the S-only model is compared against LeWM,
Fast-LeWM-style dense prefix prediction, and the existing joint successor
variant under the same data, CEM budget, and seeds.

## Group-balanced controlled variant

The lifted vector is `z / sqrt(d)`. A coordinate-wise mean squared error on
that vector therefore divides latent MSE by `d` a second time and makes the
192-dimensional direction much weaker than the scalar squared-norm feature.
The controlled `rf_balanced_successor_sequence_wm` variant changes only this
reduction:

\[
\mathcal L_{\mathrm{vec}}
=\mathbb E\left[\sum_{i=1}^{d}
(\hat S_i-S_i^\star)^2\right].
\]

The sum exactly equals latent-coordinate MSE under the `sqrt(d)` scaling. The
successor construction, online target, SIGReg, architecture, optimizer, data,
and planner remain unchanged, so a one-epoch comparison isolates this error
scaling before introducing less controlled architectural changes.

For rapid screening, evaluation accepts `--pilot`. It locks a shared budget of
10 episodes, 128 CEM candidates, 10 iterations, 16 elites, and 100 environment
steps. This mode is only a successive-halving gate; any selected checkpoint is
rerun with the unchanged 50-episode formal protocol before reporting a result.

## EMA target controlled variant

`rf_ema_balanced_successor_sequence_wm` keeps the group-balanced architecture,
loss, optimizer, data, and planner fixed, but computes future targets with a
stop-gradient EMA copy of the online encoder (`decay = 0.995`). The online
encoder still receives predictive gradients through the history branch and
SIGReg. This isolates whether moving both sides of the same prediction error
causes representation drift; the EMA network is discarded at inference.

## Direct moment controlled variant

`rf_direct_moment_sequence_wm` directly supervises every lifted future moment
against a stop-gradient online target. This removes the cumulative-prefix
attenuation of late horizons: `m_1,...,m_K` receive equal predictive weight,
while `S_1,...,S_K` are still produced by the exact discounted cumulative sum.
It remains one predictive loss plus SIGReg, with no LeWM dynamics loss, EMA
network, recurrence penalty, reward, goal, value bootstrap, or policy.

`rf_e2e_moment_sequence_wm` is the paired target-gradient control. It uses the
same direct all-horizon moment loss and inference path, but lets the prediction
error update both the history and future online-encoder branches. Comparing it
with the stop-gradient version isolates target-gradient routing without adding
another loss or changing the planner.

## Manifold prefix variant

`rf_manifold_prefix_successor_wm` removes the independently predicted
squared-norm coordinate. A causal Transformer reads the current latent and
each supplied action prefix, then a conditional residual predictor outputs
the complete future latent sequence:

\[
\hat z_{t+1:t+K}=F_\theta(z_t,a_{t:t+K-1}),\qquad
\hat m_h=\phi(\hat z_{t+h}),\qquad
\hat S_h=\frac{\sum_{j=1}^{h}\gamma^{j-1}\hat m_j}{Z_h}.
\]

Only `z` is learned. Its norm moment, constant coordinate, every one-step to
multi-step successor value, and recurrence are deterministic consequences of
the same prediction. The complete trainable objective is therefore

\[
\mathcal L=\operatorname{MSE}(\hat z_{t+1:t+K},z_{t+1:t+K})
+\lambda_{\rm SIGReg}\operatorname{SIGReg}(Z).
\]

This is the cleanest test of whether the small GRU head was the bottleneck:
data, encoder, optimizer, horizon, SIGReg, and formal CEM protocol stay fixed,
while the prefix backbone becomes substantially stronger and the successor
geometry becomes exact. Dense action-prefix latent prediction also overlaps
the backbone idea studied by [Fast-LeWM](https://arxiv.org/abs/2606.26217), so
that architecture alone is treated as a control rather than a novelty claim.

## EMA manifold target variant

`rf_ema_manifold_prefix_successor_wm` changes only the target gradient route
of the manifold-prefix model. Future latents come from a stop-gradient EMA
encoder (`decay = 0.995`), while the online encoder still learns through the
history branch and SIGReg. The Transformer, optimizer, data, horizon, planner,
and the single latent-sequence loss remain unchanged. This controlled test
targets the observed failure mode in which online validation MSE improved but
Cube planning regressed; the EMA encoder is training-only and is not exported
for planning.

## Frozen-geometry head refinement

`rf_manifold_prefix_successor_wm_cube_head_refine` resumes the online
manifold model at epoch one, freezes the complete LeWM encoder/projector in
evaluation mode, and trains only the causal action-prefix Transformer for one
additional epoch. The latent target, loss, optimizer schedule, data order, and
planner are unchanged. This isolates whether later control regression comes
from representation drift: the known epoch-one goal geometry is preserved
exactly while the dynamics head receives another full pass of supervision.

## Frozen pretrained geometry variant

`rf_frozen_manifold_prefix_successor_wm` starts from an audited pretrained
LeWM checkpoint and freezes the complete world model before the first update.
Only the causal action-prefix Transformer is trained. For each supplied action
prefix it predicts `z_{t+1},...,z_{t+K}` in the fixed LeWM coordinate system;
all finite-horizon successor quantities remain exact deterministic transforms
of those predictions.

Because the representation cannot move, SIGReg is inactive and the complete
trainable objective is the single all-horizon latent MSE. This isolates the
useful question cleanly: can dense one-step-to-multi-step prediction improve
planning while preserving the goal-distance geometry already learned by a
strong reward-free LeWM?

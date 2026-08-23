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

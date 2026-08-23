# RF-Successor-LeWM

## Status

RF-Successor-LeWM is a new method, not a renamed LS-LeWM checkpoint. It uses
the public `stable-worldmodel==0.1.1` LeWM, rollout, CEM, data, checkpoint, and
evaluation APIs. The implementation is tested, but no formal checkpoint or
controlled performance result exists yet.

## Prediction Objects

Let `H_t` be the encoded observation history and let

\[
A_{1:K}=(a_t,\ldots,a_{t+K-1})
\]

be an externally supplied candidate action prefix. LeWM predicts every future
latent in an open-loop rollout:

\[
\hat z_{t+h}=F_\theta(H_t,A_{1:h}),\qquad h=1,\ldots,K.
\]

A separate causal head predicts a finite-horizon successor for every prefix:

\[
\hat S_h=G_\eta(H_t,A_{1:h}).
\]

`G` neither receives a goal nor outputs an action. Changing action
`a_{t+j}` cannot change any `S_h` for `h < j`; the GRU architecture enforces
this causality.

## Exact Goal Geometry

For latent dimension `d`, use the fixed lift

\[
\phi(z)=\left[\frac{z}{\sqrt d},\frac{\lVert z\rVert^2}{d},1\right]
\]

and the goal query

\[
w(g)=\left[-\frac{2g}{\sqrt d},1,\frac{\lVert g\rVert^2}{d}\right].
\]

They satisfy

\[
\phi(z)^\top w(g)=\frac{\lVert z-g\rVert^2}{d}.
\]

Thus training can remain goal-free while a newly encoded goal can be queried at
planning time.

## Direct Multi-Horizon Targets

An EMA target world model encodes the future trajectory as
`bar(z)_{t+1:t+K}`. Define

\[
Z_h=\sum_{j=1}^{h}\gamma^{j-1},\qquad
Y_h=\frac{1}{Z_h}\sum_{j=1}^{h}\gamma^{j-1}
\phi(\bar z_{t+j}).
\]

This is a direct finite-horizon Monte Carlo target. It has no reward, goal,
continuation policy, TD bootstrap, or beyond-clip assumption. In particular,

\[
Y_1=\phi(\bar z_{t+1}),
\]

so the sequence contains explicit one-step through K-step supervision.

## Shared-Future Consistency

The successor and latent rollout describe the same future and must obey

\[
Z_h\hat S_h-Z_{h-1}\hat S_{h-1}
\approx \gamma^{h-1}\phi(\hat z_{t+h}),
\qquad Z_0\hat S_0=0.
\]

The complete objective is

\[
\mathcal L=\mathcal L_{\text{local LeWM}}
+\lambda_z\sum_h\lVert\hat z_{t+h}-\bar z_{t+h}\rVert^2
+\lambda_S\sum_h\lVert\hat S_h-Y_h\rVert^2
+\lambda_R\sum_h\lVert R_h\rVert^2
+\lambda_{\text{sig}}\mathcal L_{\text{SIGReg}},
\]

where `R_h` is the recurrence residual above. The successor and recurrence
losses update the online encoder and LeWM predictor; EMA targets remain
stop-gradient.

## Planning

For each CEM candidate prefix, the goal-conditioned finite-horizon cost is

\[
Q_h(H_t,A_{1:h};g)=\hat S_h^\top w(g).
\]

This is an action-sequence-conditioned `Q`, not a policy value `V`. CEM creates
the corresponding value by minimizing over supplied candidates and MPC executes
the first action block before replanning. The primary protocol uses only
`Q_K`; a terminal LeWM cost is available as an explicit configured ablation.

This feature basis supports zero-shot squared latent distance to any encoded
goal. It does not by itself support an arbitrary nonlinear reward.

## Main Risks

- CEM can propose action prefixes outside the offline action distribution.
- A cumulative goal cost means reach-and-stay, while a terminal cost means
  arrive at the final horizon; these objectives must not be conflated.
- Long open-loop losses increase compute and can dominate local representation
  learning, so the auxiliary losses use a warmup.
- First-step planning may contain less history than training; the adapter uses
  the same repeat-padding convention as the existing LeWM integrations.

## Implementation Map

- Geometry: `src/tdwm/methods/successor_geometry.py`
- Successor target, head, and loss: `src/tdwm/methods/rf_successor_lewm.py`
- Joint training and paired exports: `src/tdwm/training/rf_successor_lewm.py`
- Policy-free CEM adapter: `src/tdwm/adapters/rf_successor_lewm.py`
- Controlled evaluation: `src/tdwm/evaluation/rf_successor_lewm.py`
- Training protocol: `configs/experiment/rf_successor_lewm_cube_train.yaml`
- Evaluation protocol: `configs/experiment/rf_successor_lewm_cube_checkpoint_o50.yaml`

Run the required training smoke and resume checks before a formal run:

```bash
python scripts/train_rf_successor_lewm.py --smoke --resume never --seed 0 \
  --dataset "$TDWM_CUBE_DATASET"
python scripts/train_rf_successor_lewm.py --smoke --resume required --seed 0 \
  --dataset "$TDWM_CUBE_DATASET"
```

Evaluate paired exports from the same epoch:

```bash
python scripts/evaluate_rf_successor_lewm.py --smoke \
  --dataset "$TDWM_CUBE_DATASET" \
  --base-checkpoint-path /path/to/exports/checkpoints/epoch_10 \
  --successor-checkpoint-path /path/to/rf_successor_lewm/epoch_10.pt
```

The deployment checkpoint stores the paired base-export SHA-256, so an
independently selected LeWM checkpoint is rejected.

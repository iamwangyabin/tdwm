# 服务器实验产物索引

记录日期：2026-08-26（Asia/Shanghai）

## 范围与保存策略

本索引记录通过 SSH 从服务器读取到的实验产物。服务器侧来源为：

- `/gemini/code/tdwm/outputs/lewm_cube_training/`
- `/gemini/code/tdwm-successor-run/outputs/`

原始日志、metrics、manifest 和结果 JSON 已保存到本地被 Git 忽略的目录：

- `outputs/server_experiments/tdwm/`
- `outputs/server_experiments/tdwm_successor/`

本次本地下载目录中**没有 checkpoint 文件**（无 `.ckpt`、`.pt` 或 `.pth`）。原始大型
产物不进入 GitHub；GitHub 只保存本索引和轻量汇总。

## 当前服务器保留的原始日志

- LeWM/Cube：39 个日志文件，包含 smoke、DataLoader/Lance 读取、block shuffle、
  prefetch、compile、scheduler、正式训练和两次 CEM 评测。
- Reward-free successor sequence WM：1 个训练日志，包含 Cube 10-epoch 训练。
- 合计：40 个日志文件。

### LeWM/Cube 日志清单

相对于服务器目录 `/gemini/code/tdwm/outputs/lewm_cube_training/`：

```text
loader_w12p2_s50_epoch.log
loader_w6p1_s50_epoch.log
loader_w6p2_s100.log
loader_w6p2_s50_epoch.log
loader_w8p2_s50_epoch.log
logs/baseline_w0_20_v1.console.log
logs/blockprefetch512_w2_100.console.log
logs/blockprefetch512_w2_100_v2.console.log
logs/blockseq_w2_100.console.log
logs/blockshuffle_w0_100_profile.console.log
logs/blockshuffle_w0_20.console.log
logs/blockshuffle_w0_20_v2.console.log
logs/blockshuffle_w2_100_profile.console.log
logs/blockshuffle_w2_b192_100_profile.console.log
logs/blockshuffle_w2_b192_100_profile_v2.console.log
logs/blockshuffle_w2_pf1_500.console.log
logs/blockshuffle_w2_pf2_100.console.log
logs/blockshuffle_w2_pf2_500.console.log
logs/blockshuffle_w2_pf2_compile_100.console.log
logs/blockshuffle_w4_100_profile.console.log
logs/seed_3072_blockshuffle_w2_pf2_formal.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.evaluation.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.evaluation_cem30.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.evaluation_egl.console.log
logs/seed_3072_scheduler_epoch_w2_pf2.console.log
logs/seed_3072_smoke_optimizer_step_fixed.console.log
logs/seed_3072_smoke_scheduler_epoch_w2_pf2.console.log
logs/smoke_resume_seed_42.log
logs/smoke_seed_3072.log
logs/smoke_seed_42.log
logs/stage_cube_to_code_and_train_seed_0.log
logs/train_seed_0.log
seed_3072_episode_stream_formal_v1/stdout.log
seed_3072_formal_compiled/console-restart.log
seed_3072_formal_compiled/console.log
seed_3072_formal_eager/console.log
seed_3072_formal_eager_w0/console.log
train_seed_0_loader_w6p1_formal.log
```

### Successor sequence WM 日志

```text
/gemini/code/tdwm-successor-run/outputs/logs/rf_successor_sequence_wm_seed0.log
```

## 关键可复核结果

### LeWM Cube：seed 3072

产物目录：

`/gemini/code/tdwm/outputs/lewm_cube_training/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01/`

- 10 epochs，`global_step=127960`
- `train/loss=0.13555546`
- `train/prediction_loss=0.01771848`
- `validation/loss=0.22475678`
- `validation/prediction_loss=0.01425450`
- 50-episode、30-iteration CEM：`48.0%`
- 早期非规范 EGL 诊断：`54.0%`
- 参数量：18,034,628

对应评测结果文件位于两个 `evaluation_o25_*` 目录的 `results.json` 中。54% 只保留为
诊断结果，正式主结果是 30-iteration CEM 的 48%。

### Reward-free successor sequence WM：Cube seed 0

产物目录：

`/gemini/code/tdwm-successor-run/outputs/rf_successor_sequence_wm_cube_training/seed_0/`

- 10 epochs，`global_step=127960`
- `train/loss=0.06427395`
- `validation/loss=0.23557366`
- `train/successor_sequence_loss=0.00111796`
- successor MSE h1/hK：`0.00504898 / 0.00104171`
- 峰值 CUDA 显存：约 3.83 GiB
- 当前没有对应的 policy/CEM success-rate 结果。

## 只有报告、当前服务器没有原始日志的实验

以下记录在仓库报告中，但报告注明原始日志或 checkpoint 已清理/保存在其他外部
artifact 位置，不能当作本次下载到的 raw log：

- [`reports/mc_gt_lewm_cube_seed3072.md`](mc_gt_lewm_cube_seed3072.md)：MC-GT-LeWM
  value head，20 epochs；尚未做 CEM。
- [`reports/td_gt_lewm_cube_seed3072.md`](td_gt_lewm_cube_seed3072.md)：TD-GT-LeWM
  value head，20 epochs；尚未做 CEM。
- [`reports/e2e_joint_td_gt_lewm_cube_seed3072.md`](e2e_joint_td_gt_lewm_cube_seed3072.md)：
  E2E Joint TD-GT-LeWM，报告的 baseline/world-only/tail 结果为 72%/62%/56%。
- [`reports/baseline_tdmpc2_cartpole.md`](baseline_tdmpc2_cartpole.md)：TD-MPC2
  CartPole sparse/dense 诊断。
- [`reports/baseline_lewm_pusht_training.md`](baseline_lewm_pusht_training.md)：LeWM
  PushT，仅记录到第 0 个 epoch 的中途状态。

## 解释边界

服务器代码中还有尚未产生当前 raw output 的 experiment 配置，因此本索引只代表当前
服务器上实际保留并成功读取到的日志和轻量结果，不代表所有曾经启动过的任务或所有
研究设想均有可恢复 artifact。服务器侧 `/gemini/code/tdwm` 工作树在采集时不是干净
工作树，正式比较前仍需核对代码版本和未提交修改。

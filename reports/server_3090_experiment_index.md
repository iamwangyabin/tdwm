# 3090 服务器实验结果索引

首次记录：2026-08-26；最后更新：2026-08-27（Asia/Shanghai）

## 来源与保存策略

- SSH 来源：已配置的 `3090` 主机，项目输出目录为 `/home/yabin/tdwm/outputs/`。
- 服务器输出目录约 53G；本次只下载目录中非 `checkpoints/` 路径下的 `.log`、`.json`、
  `.jsonl` 和 `.csv` 文件。
- 本地原始轻量结果：
  `/Users/wangyabin/Documents/GitHub/tdwm/outputs/server_experiments/3090/outputs/`
- 共 459 个文件、8,198,700 bytes：42 个日志、357 个 JSON、51 个 CSV、9 个 JSONL；
  JSON 全部解析通过。
- 未下载 `.ckpt`、`.pt`、`.pth`、`.safetensors`、`.npz`、`.npy` 或任何
  `checkpoints/` 内容。原始结果目录被 Git 忽略，GitHub 只保存本索引。

## 主要训练结果

| 实验 | 训练状态 | 轻量指标/评测 |
| --- | --- | --- |
| `seed_3072_3090_episode_stream_formal` | epoch 10，`global_step=127960` | validation loss `0.6497182`，prediction loss `0.0054175`，sigreg `7.1590`；O25 评测 `72%` |
| `mc_gt_lewm_cube_full` | epoch 20，`global_step=4720` | validation MC MSE `0.00607095`；O25 评测 `74%` |
| `td_gt_lewm_cube_full` | epoch 20，`global_step=4720` | validation MC MSE `0.00864504`；O25 评测 `72%` |
| `e2e_joint_td_gt_lewm_cube_full` | epoch 10，`global_step=127960` | O25 评测 `56%` |
| `aligned_e2e_mc_gt_lewm_cube_full_v2` | epoch 10，`global_step=127960` | 正式 O50 评测 `62%`（31/50）；配对 LeWM 为 `54%`（27/50） |
| `gt_lewm_cube_training_v2` | epoch 10，`global_step=127960` | 已保留训练 manifest、结果和 metrics |
| `rf_successor_lewm_cube_v1` | epoch 10，`global_step=127960` | 已保留训练 manifest、结果和 metrics |
| `successor_geometry_lewm_cube_seed399_v2` | epoch 3，`global_step=12000` | O50 评测 `46%`；world-only `40%` |
| `policy_auxiliary_successor_geometry_lewm_cube_seed399_qtricks_v1` | epoch 3，`global_step=12000` | O50 评测 `40%` |
| `residual_policy_successor_geometry_lewm_cube_seed399_qtricks_v1` | epoch 3，`global_step=12000` | O50 评测 `32%` |

## 已保存的正式/主评测结果

下表来自下载的 `results.json`；每行均为结果文件中记录的 50 个 episode，O25/O50
表示目标偏移，不代表 episode 数。pilot 和 smoke 结果不列入主表。

| 结果文件（相对 3090 输出目录） | success rate |
| --- | ---: |
| `seed_3072_3090_episode_stream_formal/evaluation_o25/results.json` | 72% |
| `aligned_e2e_mc_gt_lewm_cube_seed3072_epoch10_o50/results.json` | 62% |
| `lewm_seed3072_e10_matched_o50_retry/results.json` | 54% |
| `mc_gt_lewm_cube_seed3072_o25/results.json` | 74% |
| `td_gt_lewm_cube_seed3072_o25/results.json` | 72% |
| `joint_td_gt_lewm_cube_seed3072_o25/results.json` | 62% |
| `e2e_joint_td_gt_lewm_cube_seed3072_o25/results.json` | 56% |
| `diagnostics/e2e_world_only_seed3072_epoch9_o25/results.json` | 62% |
| `successor_geometry_lewm_cube_seed399_v2/evaluation_epoch03_o50/results.json` | 46% |
| `successor_geometry_lewm_cube_seed399_v2/evaluation_epoch03_world_only_o50/results.json` | 40% |
| `policy_auxiliary_successor_geometry_lewm_cube_seed399_qtricks_v1/evaluation_epoch03_o50/results.json` | 40% |
| `residual_policy_successor_geometry_lewm_cube_seed399_qtricks_v1/evaluation_epoch03_o50/results.json` | 32% |
| `rf_successor_sequence_wm_cube_v1/evaluation/epoch_01_formal_o50/results.json` | 24% |
| `rf_balanced_successor_sequence_wm_cube_v1/evaluation/epoch_01_formal_o50/results.json` | 22% |
| `rf_balanced_successor_sequence_wm_cube_v1/evaluation/epoch_03_formal_o50/results.json` | 30% |
| `rf_balanced_successor_sequence_wm_cube_v1/evaluation/epoch_04_formal_o50/results.json` | 38% |
| `rf_balanced_successor_sequence_wm_cube_v1/evaluation/epoch_04_projected_formal_o50/results.json` | 34% |
| `rf_balanced_successor_sequence_wm_cube_v1/evaluation/epoch_05_formal_o50/results.json` | 36% |
| `rf_e2e_moment_sequence_wm_cube_v1/evaluation/epoch_01_formal_o50/results.json` | 30% |
| `rf_e2e_moment_sequence_wm_cube_v1/evaluation/epoch_02_formal_o50/results.json` | 34% |
| `rf_e2e_moment_sequence_wm_cube_v1/evaluation/epoch_02_projected_formal_o50/results.json` | 36% |
| `rf_e2e_moment_sequence_wm_cube_v1/evaluation/epoch_02_terminal_formal_o50/results.json` | 34% |
| `rf_e2e_moment_sequence_wm_cube_v1/evaluation/epoch_03_formal_o50/results.json` | 30% |
| `rf_ema_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_terminal_formal_o50/results.json` | 30% |
| `rf_frozen_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_path_formal_o50/results.json` | 42% |
| `rf_frozen_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_terminal_formal_o50/results.json` | 44% |
| `rf_frozen_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_validation_fit_cost_mix_formal_o50/results.json` | 40% |
| `rf_frozen_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_validation_fit_hybrid_formal_o50/results.json` | 56% |
| `rf_frozen_residual_prefix_wm_cube_lr5e5_20260824/seed_0/evaluation/epoch_01_terminal_formal_o50/results.json` | 48% |
| `rf_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_blend_formal_o50/results.json` | 38% |
| `rf_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_formal_o50/results.json` | 40% |
| `rf_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_01_terminal_formal_o50/results.json` | 44% |
| `rf_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_02_formal_o50/results.json` | 40% |
| `rf_manifold_prefix_successor_wm_cube_v1/evaluation/epoch_02_terminal_formal_o50/results.json` | 30% |
| `rf_manifold_prefix_successor_wm_cube_head_refine_v1/evaluation/epoch_02_path_formal_o50/results.json` | 38% |
| `rf_manifold_prefix_successor_wm_cube_head_refine_v1/evaluation/epoch_02_terminal_formal_o50/results.json` | 42% |

Aligned 与 `lewm_seed3072_e10_matched_o50_retry` 使用完全相同的 50 个 episode 选择和相同
的 O50 CEM 预算。配对计数为 both-success 26、Aligned-only 5、LeWM-only 1、neither 18；
双侧 exact McNemar `p=0.21875`。详见
[`aligned_e2e_mc_gt_lewm_cube_seed3072.md`](aligned_e2e_mc_gt_lewm_cube_seed3072.md)。

## 对应关系与解释边界

3090 上没有发现云服务器上的以下精确目录名：

- `seed_3072_blockshuffle_w2_pf2_valblock_resume_e01`
- `rf_successor_sequence_wm_cube_training`
- `lewm_cube_training`

因此上表是 3090 上实际存在的相近实验，不应直接当作云服务器运行的同一副本。
正式比较前仍需按各自 `protocol_manifest.json` 核对数据划分、目标偏移、规划预算、
checkpoint 版本、随机种子和代码提交版本。

# Source Index — every number → original file

All paths below are under repo root `d:/act` unless noted.  
`Week01_ACT/results/` holds **copies** for the note; prefer originals if they disagree (they should not).

---

## 1. Main Transfer Cube ($k=100$)

| Claim | Exact value | Source file |
|---|---|---|
| No TA success | 0.84 = 42/50 | `ckpts/transfer_cube/eval_no_agg/result_policy_best.txt` |
| No TA avg return | 555.24 | same |
| No TA reward ≥1/2/4 | 49/50, 45/50, 42/50 | same |
| No TA fail IDs | 7,26,30,33,38,42,45,49 | same (highest_reward row) |
| TA success | 0.98 = 49/50 | `ckpts/transfer_cube/eval_temporal_agg/result_policy_best.txt` |
| TA avg return | 604.76 | same |
| TA only fail | rollout 33 | same |
| Rollout 33 pose | x=0.009017462464891502, y=0.41886193238073793, z=0.05 | No-TA result file `box_poses:` Rollout 33; also `failure_analysis/box_poses.txt` |
| Best epoch / val loss | 1780 / 0.0213 | `ckpts/chunk_ablation/chunk_ablation_report.md` §3 (no separate train log txt in repo) |

Copies: `Week01_ACT/results/eval_k100_no_agg.txt`, `eval_k100_temporal_agg.txt`.

---

## 2. Chunk ablation

| $k$ | Mode | Success | Return | Source |
|---|---|---|---|---|
| 1 | no TA | 0.0 (0/50) | 0.0 | `ckpts/chunk_ablation/k1/eval_no_agg/result_policy_best.txt` |
| 1 | TA | 0.0 (0/50) | 0.0 | `.../k1/eval_temporal_agg/result_policy_best.txt` |
| 20 | no TA | 0.32 (16/50) | 220.8 | `.../k20/eval_no_agg/result_policy_best.txt` |
| 20 | TA | 0.64 (32/50) | **463.86** | `.../k20/eval_temporal_agg/result_policy_best.txt` |
| 50 | no TA | 0.72 (36/50) | 492.1 | `.../k50/eval_no_agg/result_policy_best.txt` |
| 50 | TA | 0.74 (37/50) | 505.3 | `.../k50/eval_temporal_agg/result_policy_best.txt` |
| 100 | — | see §1 | — | `ckpts/transfer_cube/...` |

Staged reward counts: same files’ `Reward >=` lines; also tabulated in `chunk_ablation_report.md` §4.2.

Training meta (best epoch / val loss): `chunk_ablation_report.md` §3.  
$k=20$ best val loss: **not recorded**.

Copies: `Week01_ACT/results/eval_k{1,20,50}_{no_agg,temporal_agg}.txt`, `chunk_ablation_report.md`.

Figure data hardcoded in `ckpts/chunk_ablation/make_report_figure.py` (matches the table).

---

## 3. Failure analysis

| Claim | Value | Source |
|---|---|---|
| Fixed pose 10× success | 0/10, return 0 | `ckpts/transfer_cube/fixed_pose_eval/result_policy_best_fixed_pose.txt` |
| Fixed pose coordinates | same as rollout 33 | same file `box_poses:` |
| Grid definition 7×7 | x linspace 0.005→0.089, y 0.405→0.489 | `failure_analysis/make_region_grid.py`, `region_grid_poses.csv` |
| Grid 49-cell counts | 34 success / 6 partial / 9 reward0 | `failure_region/result_policy_best_region.csv` rows 0–48; confirmed by `plot_failure_region.py` logic |
| Exact #33 replay (row 49) | reward 0 | same CSV |
| Sampling box | [0,0.2]×[0.4,0.6] | `utils.py::sample_box_pose` |

Copies: `result_policy_best_fixed_pose.txt`, `result_policy_best_region.csv`, `region_grid_poses.csv`, `box_poses.txt`.

Figures copied to `Week01_ACT/figures/`: `failure_region_map.png`, `cube_xy_scatter.png`, `video33_frame0.png`.

---

## 4. Constants used in prose

| Symbol | Value | Source |
|---|---|---|
| DT | 0.02 | `constants.py` |
| episode_len | 400 | `constants.py` SIM_TASK_CONFIGS |
| action dim | 14 | `imitate_episodes.py` `state_dim` |
| latent_dim | 32 | `detr_vae.py` |
| enc/dec/nheads | 4 / 7 / 8 | `imitate_episodes.py` |
| TA exp weight $k$ | 0.01 | `imitate_episodes.py` eval loop |

---

## 5. Rounding used in README tables

| Displayed | Exact | OK to use in README? |
|---|---|---|
| 555.2 | 555.24 | yes (with “exact” note) |
| 604.8 | 604.76 | yes |
| 463.9 | 463.86 | yes |
| x≈0.009, y≈0.419 | full float above | yes |

Subdocs prefer exact floats.

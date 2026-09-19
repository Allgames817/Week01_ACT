# 04 — Reproduction (Transfer Cube, \(k=100\))

Primary checkpoint directory: `ckpts/transfer_cube/`.  
Raw logs copied to [`results/eval_k100_no_agg.txt`](results/eval_k100_no_agg.txt) and [`results/eval_k100_temporal_agg.txt`](results/eval_k100_temporal_agg.txt).

---

## 1. Environment / task

| Item | Value |
|---|---|
| Task | `sim_transfer_cube_scripted` |
| Simulator | MuJoCo + dm_control (see repo README) |
| Cameras | `top` only |
| Action / state dim | 14 |
| Episode length | 400 |
| `DT` | 0.02 s → 8 s episodes |
| Success criterion | `highest_reward == 4` (grasp + transfer off table) |

Cube init sampling (`utils.sample_box_pose`):

\[
x \sim U[0, 0.2],\quad y \sim U[0.4, 0.6],\quad z = 0.05
\]

---

## 2. Dataset

| Item | Value |
|---|---|
| Path | `D:/act/data/sim_transfer_cube_scripted` |
| Episodes | **50** `episode_*.hdf5` |
| Collection | `record_sim_episodes.py` + scripted policy |
| Train/val | 80/20 after `set_seed(1)` in `main()` (**independent of `--seed`**) |

Normalization stats saved as `ckpts/transfer_cube/dataset_stats.pkl`.

---

## 3. Training config (main run)

| Hyperparameter | Value | Notes |
|---|---|---|
| Policy | ACT | |
| `chunk_size` / \(k\) | **100** | `num_queries=100` |
| `batch_size` | **2** | Official tip uses 8; this machine used 2 for 8GB VRAM |
| `hidden_dim` | 512 | |
| `dim_feedforward` | 3200 | |
| `kl_weight` | 10 | |
| `lr` | 1e-5 | backbone lr also 1e-5 |
| `num_epochs` | 2000 | |
| `--seed` | 0 | training RNG |
| Backbone | resnet18 | |
| enc / dec / heads | 4 / **7** / 8 | hardcoded |

Best checkpoint selected by **lowest validation loss** → `policy_best.ckpt`.

From existing ablation report (not re-parsed from a training log file this week):

| Metric | Reported |
|---|---|
| Best epoch | 1780 |
| Best val loss | 0.0213 |

Training curves (copied):

- [`figures/transfer_cube_train_val_loss.png`](figures/transfer_cube_train_val_loss.png)
- [`figures/transfer_cube_train_val_l1.png`](figures/transfer_cube_train_val_l1.png)
- [`figures/transfer_cube_train_val_kl.png`](figures/transfer_cube_train_val_kl.png)

---

## 4. Evaluation protocol

```text
eval_bc: set_seed(1000)
load policy_best.ckpt + dataset_stats.pkl
50 rollouts; optional --temporal_agg
```

Videos and logs written under:

- `ckpts/transfer_cube/eval_no_agg/`
- `ckpts/transfer_cube/eval_temporal_agg/`

(Eval save dirs avoid overwriting each other.)

---

## 5. Main results (\(k=100\))

### No temporal aggregation

Source: `ckpts/transfer_cube/eval_no_agg/result_policy_best.txt`

| Metric | Value |
|---|---|
| Success rate | **0.84** (42/50) |
| Average return | **555.24** |
| Reward ≥ 1 | 49/50 (98%) |
| Reward ≥ 2 | 45/50 (90%) |
| Reward ≥ 4 | 42/50 (84%) |

Failed rollouts (highest_reward ≠ 4): **7, 26, 30, 33, 38, 42, 45, 49**.  
Rollout **33**: return 0, highest_reward 0.

### Temporal aggregation

Source: `ckpts/transfer_cube/eval_temporal_agg/result_policy_best.txt`

| Metric | Value |
|---|---|
| Success rate | **0.98** (49/50) |
| Average return | **604.76** |
| Reward ≥ 1..4 | 49/50 (98%) |

**Only failure: rollout 33** (return 0).

### Horizon interpretation

| Mode | Prediction horizon | Replanning |
|---|---|---|
| No TA | \(100 \times 0.02 = 2\,\mathrm{s}\) | every 2 s (open-loop chunk) |
| TA | 2 s | every **0.02 s**, with exponential ensemble |

---

## 6. Checkpoints (reference only — not copied into this note)

| File | Role |
|---|---|
| `ckpts/transfer_cube/policy_best.ckpt` | used for all \(k=100\) evals |
| `ckpts/transfer_cube/policy_last.ckpt` | last epoch |
| `ckpts/transfer_cube/policy_epoch_*_seed_0.ckpt` | periodic / best-epoch dumps |

Do **not** move or overwrite these files when maintaining this note.

---

## 7. Not tested this week

- Official `batch_size=8` re-train
- Human demo dataset (`sim_transfer_cube_human`)
- Insertion task
- Real robot

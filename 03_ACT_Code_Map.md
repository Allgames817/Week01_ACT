# 03 — ACT Code Map

How the six core files connect for Transfer Cube training and evaluation. Paths relative to repo root `d:/act`.

---

## 1. End-to-end call chain

```
record_sim_episodes.py
  → data/sim_transfer_cube_scripted/episode_{i}.hdf5
  → utils.py::EpisodicDataset / load_data / get_norm_stats
  → DataLoader
  → imitate_episodes.py::main → train_bc / eval_bc
  → policy.py::ACTPolicy
  → detr/models/detr_vae.py::DETRVAE
  → detr/models/transformer.py::Transformer
  → action chunk [B, k, 14]
  → train: L1 + KL    |    eval: env.step / video / result_*.txt
```

Constants (`DT`, task configs, camera names): [`constants.py`](../constants.py).

---

## 2. File responsibilities

### `record_sim_episodes.py`

- Roll out scripted EE-space policy in `ee_sim_env`, replay joints in `sim_env`.
- Write HDF5: `/observations/qpos`, `/qvel`, `/observations/images/{cam}`, `/action`, attr `sim=True`.
- Transfer Cube: `PickAndTransferPolicy`, `num_episodes=50`, `episode_len=400`.

### `utils.py`

| Function / class | Role |
|---|---|
| `EpisodicDataset` | Sample random `start_ts`; load one observation; take actions from `start_ts:`; **pad to full episode length 400**; build `is_pad` |
| `get_norm_stats` | Mean/std of all qpos & actions over dataset |
| `load_data` | `set_seed` already called in `main` with **1** before this → 80/20 split; build train/val loaders |
| `sample_box_pose` | Uniform cube pose in \(x\in[0,0.2]\), \(y\in[0.4,0.6]\), \(z=0.05\) |
| `set_seed` | `torch` + `numpy` seeds |

**Important:** Dataset always returns actions shaped `[400, 14]`. Truncation to \(k\) happens in `ACTPolicy`, not in the Dataset.

### `imitate_episodes.py`

| Piece | Role |
|---|---|
| `main` | `set_seed(1)` first; build `policy_config` (`num_queries=chunk_size`, enc/dec layers); branch train vs `--eval` |
| `train_bc` | `set_seed(seed)` (CLI `--seed`, Week 1: **0**); epoch loop; save `policy_best.ckpt` by **min val loss** |
| `eval_bc` | `set_seed(1000)` for reproducible eval poses; load `policy_best.ckpt` + `dataset_stats.pkl`; closed-loop rollouts |
| Temporal agg | `query_frequency=1`, exponential ensemble (`k_exp=0.01`) |
| Extra flags (this fork) | `--fixed_box_pose`, `--box_pose_file`, `--eval_save_dir` for failure analysis |

### `policy.py` — `ACTPolicy`

- Builds model via `build_ACT_model_and_optimizer`.
- Train: ImageNet-normalize images; truncate actions/`is_pad` to `num_queries`; L1 + `kl_weight * KL`.
- Infer: return `a_hat` only.

### `detr/models/detr_vae.py` — `DETRVAE`

- CVAE encoder + ResNet backbone + policy Transformer + `action_head`.
- `latent_dim=32`; train sample / infer zeros. See [02_ACT_Architecture.md](02_ACT_Architecture.md).

### `detr/models/transformer.py`

- DETR-style encoder/decoder with positional encodings added inside attention.
- Prepends latent + proprio tokens to visual memory before encoding.
- Decoder uses `query_embed` of length `num_queries` (= \(k\)).

---

## 3. Tensor shapes along the train path

Assume `B=2`, `k=100`, one camera, hidden `d=512`.

| Stage | Shape |
|---|---|
| HDF5 `/action` per episode | `[400, 14]` |
| Dataset sample `action_data` (padded) | `[400, 14]` |
| After `ACTPolicy` truncate | `[B, 100, 14]` |
| Image batch | `[B, 1, 3, H, W]` |
| qpos batch | `[B, 14]` |
| CVAE \(z\) | `[B, 32]` |
| `a_hat` | `[B, 100, 14]` |

Eval path: `B=1`, unnormalize with `dataset_stats.pkl`, `env.step(target_qpos)`.

---

## 4. Seed protocol (Week 1)

| Stage | Seed | Effect |
|---|---|---|
| Start of `main` | **1** | Train/val episode index split (fixed across \(k\)) |
| Start of `train_bc` | **0** (`--seed`) | Model init + training RNG |
| Start of `eval_bc` | **1000** | Same 50 `sample_box_pose()` sequences across \(k\) |

Ablation fairness depends on this triad remaining unchanged.

---

## 5. Where results are written

| Artifact | Typical path |
|---|---|
| Checkpoints / curves | `ckpts/transfer_cube/` or `ckpts/chunk_ablation/k{1,20,50}/` |
| Eval videos + `result_policy_best.txt` | `.../eval_no_agg/` or `.../eval_temporal_agg/` |
| Fixed-pose study | `ckpts/transfer_cube/fixed_pose_eval/` |
| Region grid | `ckpts/transfer_cube/failure_region/result_policy_best_region.csv` |

This note only **copies** summary txt/csv/png into `Week01_ACT/`; originals stay in `ckpts/`.

# 06 — Chunk Size Ablation

English research note distilled from the existing Chinese report  
[`results/chunk_ablation_report.md`](results/chunk_ablation_report.md)  
(original: `ckpts/chunk_ablation/chunk_ablation_report.md`).  
All success/return numbers below are re-checked against `result_policy_best.txt` copies in [`results/`](results/).

Figure: [`figures/chunk_ablation_summary.png`](figures/chunk_ablation_summary.png)

---

## 1. Hypothesis

Under fixed data, split, architecture, and optimization, **action chunk length \(k\)** should strongly affect closed-loop Transfer Cube success. Very short chunks (especially \(k=1\)) should struggle with bimanual handover; longer chunks should help, optionally amplified by temporal aggregation (TA).

---

## 2. Experiment setup (controlled)

| Item | Value |
|---|---|
| Dataset | `D:/act/data/sim_transfer_cube_scripted` (50 scripted demos) |
| Train/val split | `main()` calls `set_seed(1)` then splits — **same for all \(k\)** |
| `--seed` | 0 |
| Policy | ACT, ResNet18, camera `top` |
| `hidden_dim` | 512 |
| `dim_feedforward` | 3200 |
| `kl_weight` | 10 |
| `batch_size` | **2** |
| `lr` | 1e-5 |
| `num_epochs` | 2000 |
| Episode / DT | 400 / 0.02 |
| Eval poses | `eval_bc` `set_seed(1000)` → **same 50 poses** for every \(k\) |

**Independent variable only:** \(k \in \{1, 20, 50, 100\}\) (`num_queries = chunk_size`).

| \(k\) | Checkpoint dir |
|---|---|
| 1 | `ckpts/chunk_ablation/k1` |
| 20 | `ckpts/chunk_ablation/k20` |
| 50 | `ckpts/chunk_ablation/k50` |
| 100 | `ckpts/transfer_cube` (**no** `chunk_ablation/k100/` directory) |

Each \(k\): evaluate `policy_best.ckpt` under `eval_no_agg/` and `eval_temporal_agg/` (50 videos + `result_policy_best.txt` each).

---

## 3. Training summary

| \(k\) | Best epoch (report) | Best val loss (report) | Train finished |
|---|---|---|---|
| 1 | 1980 | 0.0482 | yes |
| 20 | 1933 | **not recorded** in report | yes |
| 50 | 1933 | 0.0276 | yes |
| 100 | 1780 | 0.0213 | yes |

Lower validation L1/loss for larger \(k\) must **not** be read as success rate. \(k=1\) still reaches ~0.05 val loss with **0%** closed-loop success.

![k=1 loss curve](figures/k1_train_val_loss.png)

---

## 4. Main results

Exact values from result files (success = reward ≥ 4, \(n=50\)):

| \(k\) | Horizon | No TA Success | TA Success | No TA Return | TA Return | TA gain (pp) |
|---|---|---|---|---|---|---|
| 1 | 0.02 s | 0% (0/50) | 0% (0/50) | 0.0 | 0.0 | +0 |
| 20 | 0.40 s | 32% (16/50) | 64% (32/50) | 220.8 | **463.86** | **+32** |
| 50 | 1.00 s | 72% (36/50) | 74% (37/50) | 492.1 | 505.3 | +2 |
| 100 | 2.00 s | 84% (42/50) | 98% (49/50) | **555.24** | **604.76** | +14 |

README / summary figure may round returns to 463.9 / 555.2 / 604.8.

![Ablation summary](figures/chunk_ablation_summary.png)

### Trends

1. **No-TA success vs \(k\):** 0% → 32% → 72% → 84% (monotonic in this sweep).
2. **TA × \(k\) interaction is non-monotonic:** largest gain at \(k=20\); near-zero at \(k=50\); still helpful at \(k=100\); useless at \(k=1\).
3. Best single setting in this week: **\(k=100\) + TA = 98%**.

---

## 5. Staged rewards (task depth)

### No temporal aggregation

| Milestone | \(k=1\) | \(k=20\) | \(k=50\) | \(k=100\) |
|---|---|---|---|---|
| Reward ≥ 1 (contact) | 0/50 | 41/50 (82%) | 39/50 (78%) | 49/50 (98%) |
| Reward ≥ 2 (lift) | 0/50 | 37/50 (74%) | 38/50 (76%) | 45/50 (90%) |
| Reward = 4 (transfer) | 0/50 | 16/50 (32%) | 36/50 (72%) | 42/50 (84%) |

### Temporal aggregation

| Milestone | \(k=1\) | \(k=20\) | \(k=50\) | \(k=100\) |
|---|---|---|---|---|
| Reward ≥ 1 | 0/50 | 36/50 (72%) | 42/50 (84%) | 49/50 (98%) |
| Reward ≥ 2 | 0/50 | 35/50 (70%) | 41/50 (82%) | 49/50 (98%) |
| Reward = 4 | 0/50 | 32/50 (64%) | 37/50 (74%) | 49/50 (98%) |

**Reading:** \(k=20\) No-TA often reaches the cube / lifts it but fails handover; TA roughly doubles full success (16→32). \(k=50\) already replans every 50 steps without TA, so TA adds little. \(k=100\) benefits from both long plans and dense ensembling.

---

## 6. Per-\(k\) interpretation

### \(k=1\)

- 50/50 failures, return 0, never reaches reward ≥ 1.
- Videos: arm moves to a fixed pose then freezes (qualitative).
- Same eval code succeeds for \(k=100\) → not a broken eval entrypoint.
- **Possible explanation (hypothesis, not proven):** with absolute joint actions, consecutive actions are nearly equal to current qpos; CVAE encoder sees the true 1-step action and can leak it through \(z\); at inference \(z=0\), decoder collapses to a near-constant mean pose. TA cannot help because there is no multi-step sequence to ensemble.
- **What was not tested:** \(z\) statistics, encoder ablation, relative-action formulation.

### \(k=20\)

- First setting that completes the task at nontrivial rate.
- Max TA benefit in this sweep (+32 pp).

### \(k=50\)

- Strong No-TA (72%); TA ≈ flat (74%).

### \(k=100\)

- Best No-TA and best TA.
- Remaining TA failure = rollout 33 (see [05_Failure_Analysis.md](05_Failure_Analysis.md)).

---

## 7. Failed rollout IDs (aligned poses)

| Setting | Failure IDs (reward ≠ 4) |
|---|---|
| \(k=1\) no / TA | all 0–49 |
| \(k=20\) no TA | all except 1,2,3,10,13,16,17,18,23,25,28,39,41,44,46,48 |
| \(k=20\) TA | 6,7,11,12,14,21,24,30,31,32,33,35,36,37,38,40,42,49 |
| \(k=50\) no TA | 2,5,9,11,16,19,22,23,31,33,34,35,47,49 |
| \(k=50\) TA | 2,5,9,11,19,22,23,25,31,33,34,47,49 |
| \(k=100\) no TA | 7,26,30,33,38,42,45,49 |
| \(k=100\) TA | **33** |

Rollout **33** is hard across successful settings.

---

## 8. Conclusions

1. Performance differences in this sweep can be attributed to **action chunk length** (other factors controlled).
2. \(k=1\) is not “slightly worse ACT” — offline loss can look fine while closed-loop success is zero → **offline imitation loss ≠ closed-loop task performance**.
3. Temporal aggregation is an **inference-time** tool for mid/long chunks, not a fix for a collapsed \(k=1\) policy.
4. Report one headline number carefully: best here is **\(k=100\) + TA = 98%**; best No-TA is also \(k=100\) at 84%.

---

## 9. Limitations

| Gap | Status |
|---|---|
| Multiple random seeds for each \(k\) | **not tested** |
| `batch_size=8` (official) | **not tested** (all runs use 2) |
| Intermediate \(k\) (e.g. 10, 75) | **not tested** |
| Direct proof of CVAE collapse at \(k=1\) | **not tested** — hypothesis only |
| Human demos / insertion task | **not tested** |

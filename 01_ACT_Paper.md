# 01 — ACT Paper Concepts

Companion to [README.md](README.md). Focus: ideas from *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao et al., ACT) as they map to this repo and Week 1 experiments.

---

## 1. Behavior Cloning (BC)

Learn a policy by supervised imitation of expert demonstrations:

$$
\pi_\theta(a \mid o) \approx p_{\mathrm{expert}}(a \mid o)
$$

In this codebase:

- Observations $o_t = (I_t, q_t)$: top-camera image + 14-D qpos.
- Actions: 14-D joint targets (absolute joint positions + gripper).
- Expert data: 50 scripted Transfer Cube episodes in HDF5 (`data/sim_transfer_cube_scripted/`).

Training objective for ACT is not pure single-step MSE; it is **chunk L1 + KL** on a CVAE (see [02_ACT_Architecture.md](02_ACT_Architecture.md)).

---

## 2. Compounding Error

Single-step BC predicts $a_t$ from $o_t$. At test time, the robot executes $\hat{a}_t$, so next observation $o_{t+1}$ drifts from the expert distribution. Small errors accumulate over hundreds of steps.

Transfer Cube episode length = **400** steps at $\mathrm{DT}=0.02\,\mathrm{s}$ → **8 s** of closed-loop control. Compounding error is especially harmful for **bimanual handover** (right grasp → lift → left grasp), which requires coordinated multi-step timing.

**Week 1 evidence:** $k=1$ (single-step-like horizon) yields **0/50** success under both No TA and TA, despite low validation loss. Longer chunks recover performance (see [06_Chunk_Size_Ablation.md](06_Chunk_Size_Ablation.md)).

---

## 3. Action Chunking

Instead of $\pi(a_t \mid o_t)$, predict a short trajectory:

$$
\pi(a_{t:t+k-1} \mid I_t, q_t)
$$

Effects claimed in the paper (and tested here):

| Idea | Interpretation | Week 1 support |
|---|---|---|
| Effective horizon ↓ | Task of length $T$ becomes ~$T/k$ “decisions” if chunks are executed open-loop | No-TA success rises with $k$: 0→32→72→84% |
| Temporal coherence | Joint sequence planned jointly | Longer $k$ completes handover more often (staged rewards) |
| Reduced myopic greed | Chunk commits to multi-step motion | $k=20$ often touches/lifts but fails transfer without TA |

$$
\text{prediction horizon} = k \times \mathrm{DT}
$$

For $k=100$: **2.0 s** of planned joint motion per query.

---

## 4. CVAE (Conditional Variational Autoencoder)

Expert actions are often **multimodal** (multiple valid ways to grasp / time a transfer). ACT models a latent $z$:

**Train:** encode $(q_t, a_{t:t+k})$ → $\mu, \log\sigma^2$ → sample $z$; decode $(I_t, q_t, z)$ → $\hat{a}_{t:t+k}$; minimize L1 + $\lambda\,\mathrm{KL}$.

**Infer:** set $z = 0$ (prior mean), decode from observations only.

In code: `latent_dim = 32`, `kl_weight = 10` (`policy.py`, `detr_vae.py`).

**Caveat for this week:** for $k=1$, a **possible explanation** of total failure is that the decoder over-relies on $z$ while inference fixes $z=0$ (latent / train–infer mismatch). This was **not** directly measured (no $z$ histogram / ablation). Treat as hypothesis only.

---

## 5. Temporal Ensemble (Temporal Aggregation)

At inference, ACT can query the policy **every timestep** and average overlapping predictions for the current action with exponential weights (code: `k=0.01` in `imitate_episodes.py`).

| Mode | Query frequency | Execution |
|---|---|---|
| No aggregation | every $k$ steps | open-loop execute chunk |
| Temporal aggregation | every 1 step | weighted average of all predictions that cover time $t$ |

This separates:

- **Prediction horizon** = $k \times \mathrm{DT}$
- **Replanning horizon** = $\mathrm{DT}$ (with TA)

**Week 1 evidence (same poses):** TA gain is **non-monotonic** in $k$: +0 ($k=1$), **+32 pp** ($k=20$), +2 ($k=50$), +14 ($k=100$). Best overall: $k=100$ + TA = **98%**.

---

## 6. Core ACT Thesis (as used this week)

1. Chunk actions to fight compounding error on long-horizon bimanual tasks.
2. Use a Transformer (DETR-style queries) to decode a fixed-length action sequence.
3. Use a CVAE latent for multimodality during training; $z=0$ at test time.
4. Optionally temporally ensemble overlapping chunks at test time.

What this week **does not** claim as proven: that CVAE is optimal, that $z=0$ is optimal, or that ACT generalizes uniformly over the cube pose box (see failure region in [05_Failure_Analysis.md](05_Failure_Analysis.md)).

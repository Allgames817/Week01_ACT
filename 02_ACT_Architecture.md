# 02 — ACT Architecture

Maps the Week 1 mental model to this repository’s `DETRVAE` implementation. Primary sources: [`detr/models/detr_vae.py`](../detr/models/detr_vae.py), [`detr/models/transformer.py`](../detr/models/transformer.py), [`policy.py`](../policy.py).

---

## 1. High-level diagram

```
Training (actions provided)
──────────────────────────
  actions [B, k, 14] + qpos [B, 14]
       │
       ▼
  CVAE Encoder (TransformerEncoder)
  tokens: [CLS, qpos, a_1..a_k]
       │
       CLS → Linear → (μ, logvar) each [B, 32]
       │
       reparameterize → z [B, 32]
       │
       ▼
  ┌─────────────────────────────────────┐
  │ Policy path (shared train/infer)    │
  │  image [B, C, 3, H, W] → ResNet18   │
  │  qpos → Linear → proprio_input      │
  │  z    → Linear → latent_input       │
  │  concat into Transformer encoder    │
  │  query_embed [k, d] → Decoder      │
  │  action_head → a_hat [B, k, 14]     │
  └─────────────────────────────────────┘

Inference (no actions)
──────────────────────
  z := 0  (zeros [B, 32])  → same policy path → a_hat [B, k, 14]
```

---

## 2. Fixed hyperparameters (this Week 1 setup)

| Parameter | Value | Where set |
|---|---|---|
| `state_dim` / action dim | 14 | `imitate_episodes.py`, `detr_vae.py` |
| `hidden_dim` (\(d\)) | 512 | CLI `--hidden_dim` |
| `dim_feedforward` | 3200 | CLI |
| `enc_layers` (policy + CVAE encoder) | 4 | hardcoded in `imitate_episodes.py` |
| `dec_layers` | **7** | hardcoded (not DETR default 6) |
| `nheads` | 8 | hardcoded |
| `backbone` | `resnet18` | hardcoded |
| `latent_dim` | **32** | `DETRVAE.__init__` |
| `num_queries` | \(k\) = `chunk_size` | CLI `--chunk_size` |
| `kl_weight` \(\lambda\) | 10 | CLI |
| cameras | `['top']` | `constants.py` |

---

## 3. CVAE encoder path (training only)

From `DETRVAE.forward` when `actions is not None`:

| Step | Tensor shape |
|---|---|
| `actions` | `[B, k, 14]` (already truncated to `num_queries` in `ACTPolicy`) |
| `encoder_action_proj(actions)` | `[B, k, d]` |
| `encoder_joint_proj(qpos)` + unsqueeze | `[B, 1, d]` |
| `cls_embed` repeated | `[B, 1, d]` |
| concat tokens | `[B, 1+1+k, d]` → permute `[1+1+k, B, d]` |
| `encoder(...)` | same length sequence |
| take CLS (`encoder_output[0]`) | `[B, d]` |
| `latent_proj` → split | `μ [B, 32]`, `logvar [B, 32]` |
| `reparametrize` → `latent_sample` | `[B, 32]` |
| `latent_out_proj` | `[B, d]` |

Padding mask: CLS and qpos tokens are never pad; action pads come from the dataset `is_pad` truncated to length \(k\).

---

## 4. Inference latent

```python
latent_sample = torch.zeros([bs, self.latent_dim], ...)  # z = 0
latent_input = self.latent_out_proj(latent_sample)
mu = logvar = None
```

No sampling from \(\mathcal{N}(0,I)\) at test time in this implementation — **deterministic zero latent**.

---

## 5. Observation / policy path

| Input | Processing | Role |
|---|---|---|
| `image` `[B, num_cam, 3, H, W]` | ImageNet normalize in `ACTPolicy`; ResNet18 backbone; `input_proj` 1×1 conv | visual tokens |
| `qpos` `[B, 14]` | `input_proj_robot_state` → `[B, d]` | proprio token |
| `z` / latent | `latent_out_proj` → `[B, d]` | style / multimodality token |

In `Transformer.forward` (4-D image features):

1. Flatten spatial map to sequence length \(HW\).
2. Prepend **two** tokens: `[latent_input, proprio_input]` with learned `additional_pos_embed` (size 2).
3. Encoder over `[2 + HW]` tokens.
4. Decoder: `tgt = zeros_like(query_embed)`, `query_embed` shape `[k, B, d]`.
5. Output `hs` → `action_head`: Linear \(d \to 14\) → **`a_hat [B, k, 14]`**.
6. Also `is_pad_head` → `[B, k, 1]` (pad prediction; used less critically than L1 mask).

---

## 6. Loss (`ACTPolicy`)

```
all_l1 = L1(actions, a_hat)          # elementwise
l1     = mean over non-pad entries
kl     = KL(q(z|a,qpos) || N(0,I))   # total_kld
loss   = l1 + kl_weight * kl
```

Inference `__call__` without `actions` returns `a_hat` only (no loss).

---

## 7. Training vs inference (summary)

| | Training | Inference |
|---|---|---|
| Actions to CVAE encoder | yes | no |
| \(z\) | sampled from \(\mu,\log\mathrm{var}\) | **zeros** |
| Output | `a_hat`, `μ`, `logvar` + L1/KL | `a_hat` only |
| `num_queries` | \(k\) | \(k\) |

---

## 8. Temporal aggregation (execution, not architecture)

Handled in `eval_bc` (`imitate_episodes.py`), not inside `DETRVAE`:

- Without TA: query every \(k\) steps; take `all_actions[:, t % k]`.
- With TA: query every step; store predictions in `all_time_actions`; exponential weights `exp(-0.01 * age)` over all predictions covering time \(t\).

Architecture always predicts length-\(k\) chunks; TA only changes **how** those chunks are consumed.

# ACT Action Chunk Size Ablation 结果报告

- 任务：`sim_transfer_cube_scripted`（双臂搬方块）
- 日期：2026-09-18 ~ 2026-09-19
- 唯一自变量：`chunk_size` \(k \in \{1, 20, 50, 100\}\)
- 评测：每个 \(k\) 各做 **No Temporal Aggregation** 与 **Temporal Aggregation**，各 50 次 rollout

总图：`ckpts/chunk_ablation/chunk_ablation_summary.png`

![success and return vs k](chunk_ablation_summary.png)

---

## 1. 实验目的

检验 ACT 中 **action chunk 长度 \(k\)** 是否与仿真 Transfer Cube 成功率相关。在固定数据、划分、种子和其余超参的前提下，只改变 `num_queries = chunk_size`。

预期：过短的 chunk（尤其 \(k=1\)）难以完成多步双臂交接；更长的 chunk 配合 temporal aggregation 应更稳。

---

## 2. 对照设置（Controlled Experiment）

除 \(k\) 外全部固定：

| 项目 | 取值 |
|---|---|
| Dataset | `D:/act/data/sim_transfer_cube_scripted`，50 条 scripted demo |
| Train/val split | 代码里 `main()` 先 `set_seed(1)` 再划分，**不随 `--seed` 变** |
| `--seed` | 0（模型初始化与训练随机性） |
| Policy | ACT，backbone `resnet18`，相机 `top` |
| `hidden_dim` | 512 |
| `dim_feedforward` | 3200 |
| `kl_weight` | 10 |
| `batch_size` | **2**（8GB 显存无法稳定使用官方 8；k=100 实际按 2 训完，后续全部对齐） |
| `lr` | 1e-5 |
| `num_epochs` | 2000 |
| Episode length | 400 步，`DT=0.02` |
| 评测 seed | `eval_bc` 内 `set_seed(1000)`，四档 \(k\) 的 50 个 cube pose **相同** |

\(k=100\) 权重未复制，沿用已有目录 `ckpts/transfer_cube`。其余为：

```text
ckpts/chunk_ablation/k1
ckpts/chunk_ablation/k20
ckpts/chunk_ablation/k50
```

评测加载 `policy_best.ckpt`（验证损失最低的 epoch）。视频与 `result_policy_best.txt` 分目录存放，互不覆盖：

- `eval_no_agg/`：不加 `--temporal_agg`，每 \(k\) 步查询一次策略并开环执行
- `eval_temporal_agg/`：每步查询，对重叠的未来动作做指数加权平均

---

## 3. 训练

| \(k\) | ckpt 目录 | 最佳 epoch | 最佳 val loss | 训练是否完成 |
|---|---|---|---|---|
| 1 | `chunk_ablation/k1` | 1980 | 0.0482 | 是 |
| 20 | `chunk_ablation/k20` | 1933 | — | 是 |
| 50 | `chunk_ablation/k50` | 1933 | 0.0276 | 是 |
| 100 | `transfer_cube` | 1780 | 0.0213 | 是 |

说明：验证 L1/loss 随 \(k\) 变小，**不能**直接当成成功率。\(k=1\) 的 val loss 仍可降到 ~0.05，但闭环完全失败（见第 5 节）。

---

## 4. 评测主结果

成功定义为最高奖励达到环境满分 4（右爪抓起并交到左爪，方块离桌）。每格 \(n=50\)。

### 4.1 成功率与平均回报

| \(k\) | No agg 成功率 | No agg 平均回报 | Temporal agg 成功率 | Temporal agg 平均回报 | Agg 增益 |
|---|---:|---:|---:|---:|---:|
| 1 | 0%（0/50） | 0.0 | 0%（0/50） | 0.0 | +0 |
| 20 | 32%（16/50） | 220.8 | 64%（32/50） | 463.9 | **+32** |
| 50 | 72%（36/50） | 492.1 | 74%（37/50） | 505.3 | +2 |
| 100 | 84%（42/50） | 555.2 | 98%（49/50） | 604.8 | +14 |

成功率随 \(k\) 单调上升。Temporal aggregation 不是处处有效：对 \(k=20\) 帮助最大，对 \(k=50\) 几乎持平，对 \(k=100\) 仍能从 84% 抬到 98%，对 \(k=1\) 无效。

### 4.2 分阶段奖励（任务完成深度）

**No temporal aggregation**

| 做到哪一步 | \(k=1\) | \(k=20\) | \(k=50\) | \(k=100\) |
|---|---:|---:|---:|---:|
| Reward ≥ 1（碰到方块） | 0/50 | 41/50（82%） | 39/50（78%） | 49/50（98%） |
| Reward ≥ 2（举起） | 0/50 | 37/50（74%） | 38/50（76%） | 45/50（90%） |
| Reward = 4（交接成功） | 0/50 | 16/50（32%） | 36/50（72%） | 42/50（84%） |

**Temporal aggregation**

| 做到哪一步 | \(k=1\) | \(k=20\) | \(k=50\) | \(k=100\) |
|---|---:|---:|---:|---:|
| Reward ≥ 1 | 0/50 | 36/50（72%） | 42/50（84%） | 49/50（98%） |
| Reward ≥ 2 | 0/50 | 35/50（70%） | 41/50（82%） | 49/50（98%） |
| Reward = 4 | 0/50 | 32/50（64%） | 37/50（74%） | 49/50（98%） |

---

## 5. 分档解读

### \(k=1\)

两组都是 **50/50 失败，return 全 0**，连 reward≥1 都没有。视频上机械臂先挪到一个固定姿态，随后几乎不动。

这不是评测入口写错（同一套代码下 \(k=100\) 正常工作），而是 \(k=1\) + CVAE 的典型塌缩：

1. 动作是绝对关节角，下一步与当前 `qpos` 差很小，演示里 `|action−qpos|` 中位数约 0.007。
2. 训练时 encoder 能看见这 1 步真动作，decoder 靠 \(z\) 就能把 L1 压低。
3. 推理时 \(z\) 固定为 0，网络往往输出接近无条件均值的恒定目标位姿。

Temporal aggregation 在 \(k=1\) 没有可平均的未来动作序列，因此帮不上。

### \(k=20\)

开始真正做任务。无 aggregation 时 **82% 能碰到方块，但只有 32% 完成交接**：21 次卡在 reward=2（举起后交不出）。加上 temporal aggregation 后完成数从 16 翻到 32（64%）。接触率反而略降（82%→72%），失败更接近“做成或完全不动”。短 horizon 开环不够完成双臂交接，逐步重规划收益最大。

### \(k=50\)

无 aggregation 已有 **72%**，aggregation 只到 **74%**。无 aggregation 时每 50 步就会再查询一次（400 步里约 8 次），重规划频率已经高于 \(k=100\) 的每 100 步一次，所以再改成逐步集成几乎不再加分。主增益来自更长的 chunk 本身，而不是 aggregation。

### \(k=100\)

无 aggregation **84%**，aggregation **98%**。复跑无 aggregation 与最初被覆盖的那次数字一致（42/50，平均回报 555.24）。Aggregation 后只剩 **rollout 33** 失败（cube 在采样框左下角，先前固定 pose 实验 10/10 失败）。长 chunk 提供连贯计划，逐步集成修正开环漂移。

---

## 6. 失败 rollout（未达到 reward=4）

评测 pose 由 `set_seed(1000)` 决定，各 \(k\) 对齐。Rollout **33** 在所有成功过的设置里都最难（\(k=1\) 则全部失败）。

| 设置 | 失败 ID |
|---|---|
| \(k=1\) no / agg | 全部 0–49 |
| \(k=20\) no agg | 除 1,2,3,10,13,16,17,18,23,25,28,39,41,44,46,48 外 |
| \(k=20\) agg | 6,7,11,12,14,21,24,30,31,32,33,35,36,37,38,40,42,49 |
| \(k=50\) no agg | 2,5,9,11,16,19,22,23,31,33,34,35,47,49 |
| \(k=50\) agg | 2,5,9,11,19,22,23,25,31,33,34,47,49 |
| \(k=100\) no agg | 7,26,30,33,38,42,45,49 |
| \(k=100\) agg | **33** |

---

## 7. 结论

1. **性能差异可以归因于 action chunk 长度。** 在数据、划分、种子、网络宽度、学习率和 epoch 固定时，成功率随 \(k\) 单调上升：0% → 32% → 72% → 84%（无 aggregation）。
2. **\(k=1\) 不能当作“差一点的 ACT”。** 训练曲线可以收敛，闭环却完全静止；低验证损失在这里没有任务意义。
3. **Temporal aggregation 是对长/中等 chunk 的推理技巧，不是对塌缩模型的补救。** \(k=20\) 受益最大；\(k=50\) 几乎不变；\(k=100\) 从 84% 到 98%；\(k=1\) 仍为 0。
4. 若只报一个数：本设置下 **\(k=100\) + temporal aggregation = 98%** 最好；无 aggregation 时也是 \(k=100\)（84%）最好。

---

## 8. 文件索引

| 内容 | 路径 |
|---|---|
| 本报告 | `ckpts/chunk_ablation/chunk_ablation_report.md` |
| 总图 PNG/PDF | `ckpts/chunk_ablation/chunk_ablation_summary.png`（同名 `.pdf`） |
| \(k=1\) 评测 | `ckpts/chunk_ablation/k1/eval_no_agg/`、`eval_temporal_agg/` |
| \(k=20\) 评测 | `ckpts/chunk_ablation/k20/eval_no_agg/`、`eval_temporal_agg/` |
| \(k=50\) 评测 | `ckpts/chunk_ablation/k50/eval_no_agg/`、`eval_temporal_agg/` |
| \(k=100\) 评测 | `ckpts/transfer_cube/eval_no_agg/`、`eval_temporal_agg/` |
| \(k=100\) 权重 | `ckpts/transfer_cube/policy_best.ckpt` |

每个 `eval_*` 目录内为 `result_policy_best.txt` 与 `video0.mp4`–`video49.mp4`。

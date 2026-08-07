# Three architectures: MLP · LSTM · HiVT

All three share the same setup: **the input is the motion history (displacement or velocity), the output is K=6 future trajectories, and the loss is reg + cls (Winner-Takes-All)**.
They differ only in the **encoder (the body)** and the **regression loss**. That is the whole point of the comparison.

Notation: `B` = batch, `N` = number of agents in the scene, `D` = embed_dim (64 or 128), `T` = 50 history steps, `F` = 60 future steps, `K` = 6 modes.

---

## 1. MLP (rung 1: single agent, flat)

```
Input  [B, 50, 2]              focal history (disp or vel), no absolute position
   │
   ▼  ENCODER
Flatten            → [B, 100]  merge 50×2 into one vector
Linear 100→128 + ReLU          hidden layer 1
Linear 128→128 + ReLU          hidden layer 2
   │
   ▼
h  [B, 128]                    history summary
   │
   ├─► loc   : Linear 128→720 → reshape [K, B, 60, 2]     (regression, main output)
   ├─► scale : Linear 128→720 → softplus [K, B, 60, 2]    (not trained, dropped)
   └─► pi    : Linear 128→128 + ReLU → Linear 128→6 [B, 6] (classification, picks the mode)
   │
   ▼  LOSS
WTA: pick the mode closest to GT
loss = Huber(loc of best mode, y)  +  CrossEntropy(pi, soft_target)
→ backprop → AdamW + CosineLR
```
- **No:** map, agent interaction, attention, explicit time ordering.
- **Cost:** ~230K params · ~6 s/epoch · ~8-9 min full run (CPU).

---

## 2. LSTM (rung 1: single agent, sequential)

```
Input  [B, 50, 2]              same as MLP
   │
   ▼  ENCODER  (this is the only part that differs from MLP)
nn.LSTM 2→128                  reads the 50 steps SEQUENTIALLY with one shared
                              weight set, keeping a memory (cell state) through
                              the forget / input / output gates
take hn[-1]        → [B, 128]  hidden state after reading the whole sequence
   │
   ▼
h  [B, 128]
   │
   ├─► loc   : Linear 128→720 → [K, B, 60, 2]     SAME AS MLP
   ├─► scale : Linear 128→720 → [K, B, 60, 2]     (not trained)
   └─► pi    : Linear 128→128 + ReLU → 128→6       SAME AS MLP
   │
   ▼  LOSS  = Huber(loc) + CrossEntropy(pi)        SAME AS MLP
```
- Differs from MLP **only in the encoder**: the LSTM models the time sequence; the MLP flattens it in one shot.
- **Cost:** ~230K params · ~30 s/epoch (5x slower because it unrolls 50 steps) · ~40 min full run (CPU).

---

## 3. HiVT (full model: multi-agent + map + attention)

```
Input:  history of ALL agents [N, 50, 2]  +  lane vectors (MAP)  +  agent-agent / agent-lane edges
   │
   ▼  Rotate everything into each agent's own frame (rotation-invariant)
   │
   ▼  LOCAL ENCODER  (replaces the "2 Linear" of the MLP)
① AAEncoder      agent↔agent attention within a radius, AT EACH time step → [N, 50, D]
② TemporalEncoder Transformer along the 50 time steps                     → [N, D]
③ ALEncoder      agent↔lane attention (brings in the MAP)                 → [N, D]  (local embed)
   │
   ▼  GLOBAL INTERACTOR
num_global_layers attention layers among ALL agents in the scene (long-range interaction)
multihead_proj: Linear D→K·D → [N, K, D]   (global embed, one vector per mode)
   │
   ▼  MLPDecoder (uncertain=True)   ← from local_embed + global_embed
   ├─► loc   : [K, N, 60, 2]     6 future trajectories
   ├─► scale : [K, N, 60, 2]     Laplace uncertainty, TRAINED (unlike the MLP!)
   └─► pi    : [N, K]            mode probabilities
   │
   ▼  LOSS  (masked to valid agents)
WTA: pick the mode closest to GT
loss = LaplaceNLL(loc, scale, y)  +  CrossEntropy(pi, soft_target)
→ backprop → AdamW + CosineLR
```
- **What it adds over MLP/LSTM:** ① agent-agent interaction, ② temporal attention, ③ **the map (lanes)**, ④ global interaction, ⑤ rotation invariance, ⑥ **a live scale** (Laplace NLL uses both loc and scale).
- **Cost:** HiVT-64 ~2-3 days · HiVT-128 ~5 days (GPU).

---

## Quick comparison

| | MLP | LSTM | HiVT |
|---|---|---|---|
| Sees other agents? | ✗ | ✗ | ✓ (AA + Global) |
| Uses the map? | ✗ | ✗ | ✓ (AL encoder) |
| Attention? | ✗ | ✗ | ✓ |
| Time modelling | flat | sequential (LSTM) | Transformer |
| Encoder | 2 Linear | 1 LSTM | Local + Global |
| reg_loss | Huber (dead scale) | Huber (dead scale) | **Laplace NLL (live scale)** |
| 3-head output (loc/scale/pi), K=6, WTA | ✓ | ✓ | ✓ |
| Training cost | ~9 min CPU | ~40 min CPU | ~2-5 days GPU |

**What it isolates:**
- MLP → LSTM: the value of **modelling the time sequence**.
- (MLP/LSTM) → HiVT: the value of **the map + agent interaction + attention**.
- The whole ladder keeps the same measurement setup (input/output/loss framework), so the differences are **clean, fair-comparison** numbers.

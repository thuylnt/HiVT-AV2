# HiVT-AV2: HiVT ported to Argoverse 2

This repository adapts **HiVT** (Hierarchical Vector Transformer, Zhou et al., CVPR 2022), which was originally built and evaluated only on **Argoverse 1**, so that it can train and be evaluated on the newer and larger **Argoverse 2** Motion Forecasting dataset.

It was produced during an M1 internship thesis at the University of Science and Technology of Hanoi (USTH). To my knowledge these are the first HiVT results reported on the Argoverse 2 validation set.

> This is a fork / derivative of [ZikangZhou/HiVT](https://github.com/ZikangZhou/HiVT) (Apache 2.0). The core model code (local encoder, global interactor, decoder) is unchanged; the work here is the Argoverse 2 data pipeline, a small input experiment, and modernization for recent GPUs.

## What is new compared to upstream HiVT

- **Argoverse 2 data pipeline**: reads parquet trajectories and the per-scenario JSON HD map via the official `av2` API (`preprocess_av2.py`, `datasets/argoverse_v2_dataset.py`, `datamodules/argoverse_v2_datamodule.py`).
- **Longer horizon**: 50 history / 60 future steps (5 s / 6 s) instead of 20 / 30.
- **Focal-agent centering**: the scene is centered on `focal_track_id`, and each agent's local frame uses the **true AV2 heading** instead of a motion-direction estimate.
- **Lane centerlines**: computed from the two lane boundaries with `compute_midpoint_line` (AV2 does not store centerlines directly).
- **Measured velocity channel** (`x_vel`): the 2D sensor velocity, rotated into the local frame.
- **Input modes** (`--input_mode`): `displacement` (original HiVT), `velocity` (replace displacement with measured velocity), or `both` (4-channel input; displacement and velocity are rotated as separate 2D pairs).
- **Missing features**: `turn_direction` and `traffic_control` do not exist in Argoverse 2, so they are set to a constant rather than guessed.
- **Modernized for NVIDIA Blackwell** (sm_120): runs on a CUDA 12.8 build of PyTorch.

## Results (Argoverse 2 validation set, 24,988 scenarios)

| Model | Input | minADE₆ | minFDE₆ | MR₆ |
|-------|-------|:------:|:------:|:---:|
| HiVT-64  | displacement | 0.934 | 1.927 | 0.283 |
| HiVT-64  | velocity     | 1.035 | 2.035 | 0.303 |
| HiVT-64  | both         | 0.916 | 1.926 | 0.282 |
| HiVT-128 | displacement | 0.873 | 1.756 | 0.260 |
| **HiVT-128** | **both** | **0.844** | **1.702** | **0.246** |

**Finding:** measured velocity is consistently a little *worse* as a replacement for displacement, but a little *better* when added on top of it. So velocity cannot stand alone, yet it carries some complementary information.

> Metrics follow HiVT's own convention (best of 6 modes chosen by FDE, then both ADE and FDE reported for that mode), computed on the validation set with a single seed. They are internally consistent for comparing these models, and are **not** official leaderboard (test-set, brier-minFDE) numbers.

## Repository layout

```
preprocess_av2.py                 # raw AV2 (parquet + JSON map) -> HiVT .pt graphs
train_av2.py                      # train HiVT on AV2 (--input_mode)
eval_av2.py                       # evaluate a checkpoint on the full AV2 val set
datasets/argoverse_v2_dataset.py  # AV2 dataset -> TemporalData
datamodules/argoverse_v2_datamodule.py
models/                           # HiVT model (unchanged from upstream)
losses/  metrics/  utils.py       # loss, metrics, helpers

# baselines (the "ladder" used to interpret the HiVT numbers)
baseline_cv.py                    # rule-based: constant position / velocity (K=1)
cache_focal.py                    # extract the focal agent into a compact tensor cache
models/simple_baseline.py         # single-agent MLP / LSTM (K=6, no map, no interaction)
train_simple.py                   # train the MLP / LSTM baselines on the focal cache
architectures.md                  # notes on the MLP / LSTM / HiVT architectures
viz_av2.py                        # qualitative visualization of predictions
```

## Setup

Requires Python 3.9+, a CUDA build of PyTorch, and:

```
pip install torch pytorch-lightning torch-geometric numpy
pip install av2            # official Argoverse 2 API
```

On NVIDIA Blackwell (RTX 50-series, sm_120) install a CUDA 12.8 (cu128) build of PyTorch and matching `torch-scatter` / `torch-geometric`.

## Usage

**1. Preprocess** the raw Argoverse 2 Motion Forecasting data into HiVT graphs (one `.pt` per scenario). Run once per split:

```bash
python preprocess_av2.py --data_root <av2_motion_forecasting_root> --split train
python preprocess_av2.py --data_root <av2_motion_forecasting_root> --split val
```

**2. Train** (choose the input representation):

```bash
# displacement (faithful port) | velocity | both
python train_av2.py \
  --train_dir <processed_train_dir> --val_dir <processed_val_dir> \
  --input_mode displacement

# larger model (HiVT-128)
python train_av2.py --train_dir ... --val_dir ... --input_mode both --embed_dim 128
```

**3. Evaluate** a checkpoint on the full validation set:

```bash
python eval_av2.py --ckpt <best.ckpt> --val_dir <processed_val_dir> \
  --input_mode displacement
```

## Baselines (the ladder)

To make the HiVT numbers meaningful, this repo also includes a ladder of simpler models, each adding one capability so the gap between rungs shows what that capability is worth. All of them are scored on the focal agent with the same metric code as HiVT.

```bash
# rung 0: rule-based (no training), constant position / velocity
python baseline_cv.py --val_dir <processed_val_dir>

# rung 1: build a compact focal-only cache (per split), then train single-agent MLP / LSTM
python cache_focal.py  --in_dir <processed_train_dir> --out train_focal.pt
python cache_focal.py  --in_dir <processed_val_dir>   --out val_focal.pt
python train_simple.py --train_focal train_focal.pt --val_focal val_focal.pt \
  --encoder mlp --input displacement --max_epochs 50
```

The single-agent baselines share HiVT's output design (K = 6 modes, winner-takes-all training) so that only the map and the agent interaction differ, keeping the comparison fair.

## Attribution and license

- Based on **HiVT**: Zikang Zhou et al., *HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction*, CVPR 2022 ([code](https://github.com/ZikangZhou/HiVT), Apache 2.0).
- Argoverse 2 extraction adapted from **Forecast-MAE**: Cheng et al., 2023 ([code](https://github.com/jchengai/forecast-mae)).
- Argoverse 2 dataset: Wilson et al., *Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting*, NeurIPS 2021.

Released under the **Apache License, Version 2.0** (see `LICENSE` and `NOTICE`).

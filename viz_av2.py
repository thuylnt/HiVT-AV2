"""Qualitative visualization of HiVT on Argoverse 2 (my own model, not the original AV1 checkpoint).
2x2 grid: lanes (grey) + focal history (white) + ground-truth future (green) + 6 predicted modes (thicker line = higher probability). 
Used for the thesis.

Run:
  python viz_av2.py --ckpt lightning_logs/version_9/checkpoints/epoch=47-step=479808.ckpt \
      --val_dir <hivt-av2/val> --input_mode both --scan 400 --out demo_out_av2/grid_best.png
"""
import os, glob, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_orig = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _load

from torch_geometric.data import Batch
from utils import TemporalData          
from models.hivt import HiVT

HIST, FUT = 50, 60


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def prep(data, input_mode):
    """Replicate ArgoverseV2Dataset: build data.x for the chosen input mode."""
    if input_mode == "velocity":
        data.x = data.x_vel
    elif input_mode == "both":
        data.x = torch.cat([data.x, data.x_vel], dim=-1)
    # displacement -> data.x unchanged
    return data


@torch.no_grad()
def predict(model, data, device):
    batch = Batch.from_data_list([data]).to(device)
    y_hat, pi = model(batch)               # y_hat [6, N, 60, 4], pi [N, 6]
    return y_hat.cpu(), pi.cpu()


def focal_future_scene(data):
    """GT future of focal (node 0) in scene frame = pos49 + y (y is scene-frame disp)."""
    pos49 = data.positions[0, HIST - 1].numpy()
    return pos49 + data.y[0].numpy()       # [60, 2]


def modes_scene(y_hat, data):
    """6 predicted modes of focal -> scene frame. loc is in focal local frame."""
    ang = float(data.rotate_angles[0])
    R = rot(ang)
    pos49 = data.positions[0, HIST - 1].numpy()
    return [y_hat[k, 0, :, :2].numpy() @ R.T + pos49 for k in range(y_hat.shape[0])]


def lane_segments(data):
    """Reconstruct lane segments near the focal from lane_actor_vectors (scene frame)."""
    if data.lane_actor_index.numel() == 0:
        return []
    seg_idx = data.lane_actor_index[0].numpy()
    act_idx = data.lane_actor_index[1].numpy()
    la_vec = data.lane_actor_vectors.numpy()
    lv = data.lane_vectors.numpy()
    pos49 = data.positions[0, HIST - 1].numpy()
    segs = []
    for s, a, v in zip(seg_idx, act_idx, la_vec):
        if a != 0:
            continue
        start = v + pos49                  # lane point in scene frame
        end = start + lv[s]                # + segment vector
        segs.append(np.stack([start, end]))
    return segs


def turning(data):
    """How much the focal turns over its GT future (to pick interesting scenes)."""
    fut = focal_future_scene(data)
    d = np.diff(fut, axis=0)
    plen = float(np.linalg.norm(d, axis=1).sum())
    if plen < 12.0:                        # near-stationary: skip
        return -1.0, plen
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.abs(np.diff(np.unwrap(ang))).sum()), plen


def draw(ax, data, modes, probs, title):
    ax.set_facecolor("#ffffff")
    for seg in lane_segments(data):
        ax.plot(seg[:, 0], seg[:, 1], color="#c4c8cf", lw=1.0, zorder=1)
    N = data.num_nodes
    pad = data.padding_mask
    # other agents' history
    for j in range(1, N):
        m = ~pad[j, :HIST]
        h = data.positions[j, :HIST][m].numpy()
        if len(h) > 1:
            ax.plot(h[:, 0], h[:, 1], color="#98a0ab", lw=1.1, alpha=0.8, zorder=2)
            ax.scatter(h[-1, 0], h[-1, 1], color="#98a0ab", s=12, zorder=2)
    # focal history + GT future
    ah = data.positions[0, :HIST].numpy()
    gt = focal_future_scene(data)
    ax.plot(ah[:, 0], ah[:, 1], color="#111111", lw=2.4, zorder=5, label="Focal history")
    ax.plot(np.r_[ah[-1:, 0], gt[:, 0]], np.r_[ah[-1:, 1], gt[:, 1]],
            color="#2ca02c", lw=2.8, zorder=6, label="Ground truth")
    ax.scatter(ah[-1, 0], ah[-1, 1], facecolor="#ffffff", s=55, zorder=7, ec="#111111", lw=1.3)
    # 6 modes: fixed palette by probability rank (identical colours in every panel)
    # thickness scales with probability, red = most likely.
    PALETTE = ["#d62728", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
    rank = np.empty(len(probs), int)
    rank[np.argsort(-probs)] = np.arange(len(probs))
    for k in np.argsort(probs):
        col = PALETTE[rank[k]]
        ax.plot(np.r_[ah[-1:, 0], modes[k][:, 0]], np.r_[ah[-1:, 1], modes[k][:, 1]],
                color=col, lw=1.8 + 3.2 * probs[k], alpha=0.95, zorder=4)
        ax.scatter(modes[k][-1, 0], modes[k][-1, 1], color=col, s=25 + 85 * probs[k],
                   zorder=4, ec="#333333", lw=0.4)
    focus = np.concatenate([ah, gt] + list(modes), axis=0)
    half = max(np.ptp(focus[:, 0]), np.ptp(focus[:, 1])) / 2 + 12
    mx, my = focus[:, 0].mean(), focus[:, 1].mean()
    ax.set_xlim(mx - half, mx + half)
    ax.set_ylim(my - half, my + half)
    ax.set_title(title, color="#222222", fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(colors="#666666")
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--input_mode", choices=["displacement", "velocity", "both"], default="both")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--scan", type=int, default=400)
    ap.add_argument("--out", default="demo_out_av2/grid_best.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = HiVT.load_from_checkpoint(args.ckpt, parallel=True).to(device)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.val_dir, "*.pt")))[: args.scan]
    print(f"scanning {len(files)} scenes for the widest mode fan-out...")
    scored = []
    for f in files:
        try:
            d = prep(torch.load(f), args.input_mode)
            if d.y is None:
                continue
            _, plen = turning(d)
            if plen < 12.0:                # skip near-stationary
                continue
            y_hat, _ = predict(model, d, device)
            modes = modes_scene(y_hat, d)
            ends = np.stack([m[-1] for m in modes])        # [6, 2] endpoints
            spread = float(np.linalg.norm(ends - ends.mean(0), axis=1).max())
            scored.append((spread, f))
        except Exception:
            continue
    scored.sort(reverse=True)
    # widest fan-out first (model most uncertain -> visually the multimodal "fan");
    # 4th slot = a confident, tight case for contrast
    picks = [scored[0][1], scored[1][1], scored[2][1], scored[-1][1]]
    print("mode spreads:", [round(scored[i][0], 1) for i in (0, 1, 2, -1)])
    print("picked:", [os.path.basename(p)[:12] for p in picks])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#ffffff")
    for ax, f in zip(axes.ravel(), picks):
        d = prep(torch.load(f), args.input_mode)
        y_hat, pi = predict(model, d, device)
        probs = torch.softmax(pi[0], dim=-1).numpy()
        modes = modes_scene(y_hat, d)
        gt = focal_future_scene(d)
        fde = min(float(np.linalg.norm(m[-1] - gt[-1])) for m in modes)
        draw(ax, d, modes, probs, f"minFDE = {fde:.2f} m")
    axes[0, 0].legend(loc="upper left", fontsize=8, facecolor="#ffffff",
                      labelcolor="#222222", edgecolor="#bbbbbb", framealpha=0.95)
    fig.suptitle("HiVT-128 (displacement+velocity)  |  Argoverse 2 val  |  6 predicted modes "
                 "(thicker line = higher probability; colours rank the modes)", color="#222222", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=140, facecolor="#ffffff")
    print("saved", args.out)


if __name__ == "__main__":
    main()

"""Non-learned (rule-based) baselines to compare against HiVT on AV2. 
Scored on the FOCAL agent, the same way HiVT scores (metrics/ADE,FDE,MR; MR = FDE>2m). No GPU, no training.

3 baselines (all predict the focal's 60 future steps, compared to data.y[0]):
  - CP      Constant Position: stay at the current position (absolute floor).
  - CV-disp Constant Velocity: extrapolate the last DISPLACEMENT (pos[49]-pos[48]), HiVT-A style.
  - CV-vel  Constant Velocity: extrapolate the AV2 measured VELOCITY at step 49 (m/s * dt), HiVT-B style.

Comparing CV-disp vs CV-vel already shows, at the naive level, whether displacement or velocity fits the target better.
Run FROM the HiVT-AV2 folder:
  python baseline_cv.py --val_dir /mnt/.../hivt-av2/val [--limit N]
"""
import os, glob, argparse
import torch
from tqdm import tqdm
from utils import TemporalData          # needed for torch.load of the .pt files
from metrics import ADE, FDE, MR

HIST, FUT, DT = 50, 60, 0.1
STEPS = torch.arange(1, FUT + 1).float().unsqueeze(-1)   # [60,1] = (t+1) for t=0..59


def preds_for_focal(data):
    """Return dict {name: pred[60,2]} for focal = node 0, in the scene frame (matches data.y)."""
    pos = data.positions[0]                 # [50,2] focal history positions (rotated)
    p49, p48 = pos[HIST - 1], pos[HIST - 2]
    v49 = data.x_vel[0, HIST - 1]           # velocity measured at step 49 (rotated), m/s
    return {
        "CP (stay still)":       torch.zeros(FUT, 2),
        "CV-disp (last disp)":   (p49 - p48).unsqueeze(0) * STEPS,    # (pos49-pos48)*(t+1)
        "CV-vel (AV2 velocity)": v49.unsqueeze(0) * STEPS * DT,       # v*(t+1)*dt
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.val_dir, "*.pt")))
    if args.limit:
        files = files[: args.limit]
    names = list(preds_for_focal_dummy())
    M = {n: (ADE(), FDE(), MR()) for n in names}

    skipped = 0
    for f in tqdm(files, desc="baseline"):
        data = torch.load(f, weights_only=False)
        # focal must be valid at steps 48+49 (AV2 focal always is; guard just in case)
        if bool(data.padding_mask[0, HIST - 1]) or bool(data.padding_mask[0, HIST - 2]):
            skipped += 1
            continue
        y = data.y[0].unsqueeze(0)          # [1,60,2] focal ground truth
        for name, pred in preds_for_focal(data).items():
            ade, fde, mr = M[name]
            p = pred.unsqueeze(0)           # [1,60,2]
            ade.update(p, y); fde.update(p, y); mr.update(p, y)

    print(f"\n{len(files) - skipped} scenes scored (skipped {skipped}). Scored on FOCAL, K=1.\n")
    print(f"{'Baseline':<24}{'minADE':>9}{'minFDE':>9}{'MR':>8}")
    print("-" * 50)
    for name in names:
        ade, fde, mr = M[name]
        print(f"{name:<24}{ade.compute():>9.3f}{fde.compute():>9.3f}{mr.compute():>8.3f}")
    print("\nFor reference, HiVT (K=6, full val): A disp 0.934/1.927/0.283 | B vel 1.035/2.035/0.303")


def preds_for_focal_dummy():
    """Just to get the baseline names without needing data."""
    return {"CP (stay still)": None, "CV-disp (last disp)": None, "CV-vel (AV2 velocity)": None}


if __name__ == "__main__":
    main()

"""Extract the FOCAL (node 0) from each scene .pt 
-> one compact tensor file, so the small models (1-agent MLP/LSTM) train very fast from RAM.

Saves: x_disp [M,50,2] (history displacement), x_vel [M,50,2] (measured velocity),
       y [M,60,2] (focal future). Same scene frame HiVT scores in.
Run FROM HiVT-AV2:
  python cache_focal.py --in_dir <hivt-av2/train> --out <train_focal.pt>
  python cache_focal.py --in_dir <hivt-av2/val>   --out <val_focal.pt>
"""
import os, glob, argparse
import torch
from tqdm import tqdm
from utils import TemporalData

HIST = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.pt")))
    xs, xv, ys = [], [], []
    skipped = 0
    for f in tqdm(files, desc=os.path.basename(args.out)):
        d = torch.load(f, weights_only=False)
        if bool(d.padding_mask[0, HIST - 1]) or d.y is None:
            skipped += 1
            continue
        xs.append(d.x[0].clone())          # [50,2] focal displacement
        xv.append(d.x_vel[0].clone())      # [50,2] focal velocity
        ys.append(d.y[0].clone())          # [60,2] focal future
    out = {"x_disp": torch.stack(xs), "x_vel": torch.stack(xv), "y": torch.stack(ys)}
    torch.save(out, args.out)
    print(f"{out['x_disp'].shape[0]} focal saved to {args.out} (skipped {skipped}). "
          f"x_disp{tuple(out['x_disp'].shape)} y{tuple(out['y'].shape)}")


if __name__ == "__main__":
    main()

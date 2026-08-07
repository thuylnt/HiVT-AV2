"""Eval HiVT-AV2 on the full official val set (24,988) to get numbers to compare with the paper.
Run:  python eval_av2.py --ckpt <best.ckpt> --val_dir <hivt-av2/val> --input_mode displacement|velocity
"""
from argparse import ArgumentParser

import torch
# torch>=2.6 defaults weights_only=True -> force False to load the PL checkpoint
_orig = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _load

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets.argoverse_v2_dataset import ArgoverseV2Dataset
from models.hivt import HiVT

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--input_mode", choices=["displacement", "velocity", "both"], default="displacement")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    model = HiVT.load_from_checkpoint(args.ckpt, parallel=True)   # restores hparams (50/60...)
    ds = ArgoverseV2Dataset(args.val_dir, args.input_mode)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    trainer.validate(model, dl)

"""Train the small 1-agent models (MLP/LSTM) on the focal cache: fast, no GPU contention with HiVT.
  python train_simple.py --train_focal train_focal.pt --val_focal val_focal.pt --encoder mlp --input displacement --max_epochs 50
"""
from argparse import ArgumentParser
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.simple_baseline import SimpleForecaster


def load(path, input_mode):
    d = torch.load(path, weights_only=False)
    x = d["x_disp"] if input_mode == "displacement" else d["x_vel"]
    return TensorDataset(x, d["y"])


if __name__ == "__main__":
    pl.seed_everything(2022)
    p = ArgumentParser()
    p.add_argument("--train_focal", required=True)
    p.add_argument("--val_focal", required=True)
    p.add_argument("--encoder", choices=["mlp", "lstm"], default="mlp")
    p.add_argument("--input", choices=["displacement", "velocity"], default="displacement")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpus", type=int, default=0)      # CPU by default 
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args()

    tr = load(args.train_focal, args.input)
    va = load(args.val_focal, args.input)
    print(f"[simple {args.encoder}/{args.input}] train {len(tr)} | val {len(va)}")
    trdl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    vadl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleForecaster(encoder=args.encoder, hidden_dim=args.hidden_dim,
                             lr=args.lr, T_max=args.max_epochs)
    ckpt = ModelCheckpoint(monitor="val_minADE", save_top_k=1, mode="min")
    acc = "gpu" if args.gpus > 0 else "cpu"
    trainer = pl.Trainer(accelerator=acc, devices=(args.gpus or 1), max_epochs=args.max_epochs,
                         callbacks=[ckpt], logger=pl.loggers.CSVLogger("lightning_logs_simple", name=f"{args.encoder}_{args.input}"))
    trainer.fit(model, trdl, vadl)
    print("BEST val_minADE ckpt:", ckpt.best_model_path, "=", ckpt.best_model_score.item())

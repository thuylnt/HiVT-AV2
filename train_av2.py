"""Train HiVT on Argoverse 2 (ported). Differs from original HiVT: 50 history steps / 60 future steps, 
AV2 datamodule, and a --input_mode {displacement|velocity} flag to choose between self-computed displacement and AV2's directly-measured 2D velocity."""
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules.argoverse_v2_datamodule import ArgoverseV2DataModule
from models.hivt import HiVT

if __name__ == "__main__":
    pl.seed_everything(2022)
    p = ArgumentParser()
    # data
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--input_mode", choices=["displacement", "velocity", "both"], default="displacement")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    # trainer
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--max_epochs", type=int, default=64)
    p.add_argument("--monitor", type=str, default="val_minFDE")
    p.add_argument("--save_top_k", type=int, default=5)
    # model (AV2 defaults: 50/60)
    p.add_argument("--historical_steps", type=int, default=50)
    p.add_argument("--future_steps", type=int, default=60)
    p.add_argument("--num_modes", type=int, default=6)
    p.add_argument("--rotate", type=bool, default=True)
    p.add_argument("--node_dim", type=int, default=2)
    p.add_argument("--edge_dim", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_temporal_layers", type=int, default=4)
    p.add_argument("--num_global_layers", type=int, default=3)
    p.add_argument("--local_radius", type=float, default=50)
    p.add_argument("--parallel", type=bool, default=False)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--T_max", type=int, default=64)
    p.add_argument("--resume", type=str, default=None, help=".ckpt path to resume from (restores epoch/optimizer)")
    args = p.parse_args()

    # variant C "both": x = concat(displacement, velocity) = 4 channels -> force node_dim=4
    if args.input_mode == "both" and args.node_dim != 4:
        print(f"[both] input_mode=both -> force node_dim {args.node_dim}->4 (displacement 2D + velocity 2D)")
        args.node_dim = 4

    model_ckpt = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode="min")
    trainer = pl.Trainer(accelerator="gpu", devices=args.gpus, max_epochs=args.max_epochs,
                         callbacks=[model_ckpt])
    model = HiVT(historical_steps=args.historical_steps, future_steps=args.future_steps,
                 num_modes=args.num_modes, rotate=args.rotate, node_dim=args.node_dim,
                 edge_dim=args.edge_dim, embed_dim=args.embed_dim, num_heads=args.num_heads,
                 dropout=args.dropout, num_temporal_layers=args.num_temporal_layers,
                 num_global_layers=args.num_global_layers, local_radius=args.local_radius,
                 parallel=args.parallel, lr=args.lr, weight_decay=args.weight_decay, T_max=args.T_max)
    datamodule = ArgoverseV2DataModule(
        train_dir=args.train_dir, val_dir=args.val_dir, input_mode=args.input_mode,
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        num_workers=args.num_workers)
    if args.resume:
        print(f"[resume] continue from {args.resume}")
    trainer.fit(model, datamodule, ckpt_path=args.resume)

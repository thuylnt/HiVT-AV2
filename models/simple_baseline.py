"""Small 1-agent model to compare against HiVT (the first rung of the 'baseline ladder').
Looks only at the focal's own HISTORY -> predicts 60 future steps. No map, no agent interaction, no attention, no rotation. 
Keeps K=6 + winner-takes-all + the same soft-target classification loss and ADE/FDE/MR metrics as HiVT, for a fair comparison
(the gap of A/B/C vs this = the value of map+interaction+multi-mode).

Swappable encoder: 'mlp' (direct regression) or 'lstm' (seq2seq). 
The regression loss is a Huber loss on the position, 
not HiVT's Laplace NLL: a small model tends to inflate the Laplace scale instead of improving the position, so Huber keeps the baseline stronger.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import SoftTargetCrossEntropyLoss
from metrics import ADE, FDE, MR


class SimpleForecaster(pl.LightningModule):
    def __init__(self, encoder="mlp", historical_steps=50, future_steps=60, num_modes=6,
                 input_dim=2, hidden_dim=128, lr=5e-4, weight_decay=1e-4, T_max=50, **kw):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_type = encoder
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.lr, self.weight_decay, self.T_max = lr, weight_decay, T_max

        if encoder == "mlp":
            self.enc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(historical_steps * input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        elif encoder == "lstm":
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(encoder)

        # multi-mode output: loc(2)+scale(2) per step, + logit to pick the mode
        self.loc = nn.Linear(hidden_dim, num_modes * future_steps * 2)
        self.scale = nn.Linear(hidden_dim, num_modes * future_steps * 2)
        self.pi = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, num_modes))

        self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")
        self.minADE, self.minFDE, self.minMR = ADE(), FDE(), MR()

    def forward(self, x):                       # x: [B, T, C]
        if self.encoder_type == "mlp":
            h = self.enc(x)
        else:
            _, (hn, _) = self.lstm(x)
            h = hn[-1]                          # [B, hidden]
        B = x.size(0)
        # Linear gives [B, K*60*2] -> view [B,K,60,2] (row-major correct) -> permute [K,B,60,2]
        loc = self.loc(h).view(B, self.num_modes, self.future_steps, 2).permute(1, 0, 2, 3)
        scale = F.softplus(self.scale(h)).view(B, self.num_modes, self.future_steps, 2).permute(1, 0, 2, 3)
        y_hat = torch.cat([loc, scale], dim=-1)  # [K, B, 60, 4]
        pi = self.pi(h)                          # [B, K]
        return y_hat, pi

    def _step(self, batch):
        x, y = batch                             # x[B,T,C], y[B,60,2]
        y_hat, pi = self(x)
        l2 = torch.norm(y_hat[..., :2] - y, p=2, dim=-1).sum(dim=-1)   # [K, B]
        best = l2.argmin(dim=0)
        y_hat_best = y_hat[best, torch.arange(y.size(0))]              # [B,60,4]
        return y_hat, pi, y_hat_best, l2, y

    def training_step(self, batch, _):
        y_hat, pi, y_hat_best, l2, y = self._step(batch)
        # Huber directly on loc (best mode) to avoid the loc-underfit problem of Laplace NLL
        reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        soft_target = F.softmax(-l2 / self.future_steps, dim=0).t().detach()
        cls_loss = self.cls_loss(pi, soft_target)
        loss = reg_loss + cls_loss
        self.log("train_reg_loss", reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, _):
        y_hat, pi, _, _, y = self._step(batch)
        # pick the best mode by FDE (same as HiVT)
        fde = torch.norm(y_hat[:, :, -1, :2] - y[:, -1], p=2, dim=-1)   # [K,B]
        best = fde.argmin(dim=0)
        y_best = y_hat[best, torch.arange(y.size(0))][..., :2]          # [B,60,2]
        self.minADE.update(y_best, y); self.minFDE.update(y_best, y); self.minMR.update(y_best, y)
        self.log("val_minADE", self.minADE, prog_bar=True, on_epoch=True, batch_size=y.size(0))
        self.log("val_minFDE", self.minFDE, prog_bar=True, on_epoch=True, batch_size=y.size(0))
        self.log("val_minMR", self.minMR, prog_bar=True, on_epoch=True, batch_size=y.size(0))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max, eta_min=0.0)
        return [opt], [sch]

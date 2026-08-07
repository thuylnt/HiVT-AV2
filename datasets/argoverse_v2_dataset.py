"""Dataset that reads the HiVT-AV2 cache (.pt TemporalData created by preprocess_av2.py).

input_mode picks the model input:
  - "displacement" : x = self-computed displacement (same as original HiVT)   -> node_dim 2
  - "velocity"     : x = AV2's directly-measured 2D velocity (experiment)      -> node_dim 2
  - "both"         : x = concat(displacement, velocity) = 4 channels           -> node_dim 4
"""
import os
from typing import Optional, Callable

import torch
from torch_geometric.data import Dataset


class ArgoverseV2Dataset(Dataset):
    def __init__(self, data_dir: str, input_mode: str = "displacement",
                 transform: Optional[Callable] = None):
        assert input_mode in ("displacement", "velocity", "both")
        self.data_dir = data_dir
        self.input_mode = input_mode
        self._files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
        super().__init__(transform=transform)
        print(f"[AV2 {input_mode}] {data_dir}: {len(self._files)} scenes")

    def len(self) -> int:
        return len(self._files)

    def get(self, idx):
        data = torch.load(os.path.join(self.data_dir, self._files[idx]), weights_only=False)
        if self.input_mode == "velocity":
            data.x = data.x_vel            # replace displacement with measured velocity
        elif self.input_mode == "both":
            data.x = torch.cat([data.x, data.x_vel], dim=-1)   # [N,50,2]+[N,50,2] -> [N,50,4]
        return data

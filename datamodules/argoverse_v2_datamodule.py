"""DataModule for HiVT-AV2. Points train/val at separate cache folders."""
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from datasets.argoverse_v2_dataset import ArgoverseV2Dataset


class ArgoverseV2DataModule(LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, input_mode: str = "displacement",
                 train_batch_size: int = 32, val_batch_size: int = 32, shuffle: bool = True,
                 num_workers: int = 8, pin_memory: bool = True, persistent_workers: bool = True):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.input_mode = input_mode
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV2Dataset(self.train_dir, self.input_mode)
        self.val_dataset = ArgoverseV2Dataset(self.val_dir, self.input_mode)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod

class BaseDataModule(ABC):
    def __init__(self, batch_size=32, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage=None):
        """Load and preprocess datasets"""
        pass

    @abstractmethod
    def _build_dataset(self, split):
        """Build one dataset split (train/val/test)"""
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True,drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True,drop_last=True)

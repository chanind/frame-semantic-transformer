from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from frame_semantic_transformer.constants import DEFAULT_NUM_WORKERS


class TrainDataModule(pl.LightningDataModule):
    """
    Based on https://github.com/Shivanandroy/simpleT5/blob/main/simplet5/simplet5.py
    """

    batch_size: int
    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    test_dataset: Optional[Dataset[Any]]
    num_workers: int

    def __init__(
        self,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
        test_dataset: Optional[Dataset[Any]] = None,
        batch_size: int = 8,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        dataloader: Any = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        dataset = self.test_dataset if self.test_dataset else self.val_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

from __future__ import annotations
from collections import defaultdict
import logging
import os
import argparse
from typing import Any, Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.base import Callback
from frame_semantic_transformer.constants import MODEL_MAX_LENGTH

from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
from frame_semantic_transformer.data.data_utils import trim_batch
from frame_semantic_transformer.data.load_framenet_samples import (
    load_sesame_train_samples,
    load_sesame_test_samples,
    load_sesame_dev_samples,
)
from frame_semantic_transformer.evaluate import calc_eval_metrics, evaluate_batch

DEFAULT_NUM_WORKERS = os.cpu_count() or 2
logger = logging.getLogger(__name__)


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


class TrainingModelWrapper(pl.LightningModule):
    """
    Based on https://github.com/Shivanandroy/simpleT5/blob/main/simplet5/simplet5.py
    """

    lr: float
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    trainer: pl.Trainer
    output_dir: str
    save_only_last_epoch: bool
    skip_initial_epochs_validation: int

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        lr: float = 1e-4,
        output_dir: str = "outputs",
        save_only_last_epoch: bool = False,
        skip_initial_epochs_validation: int = 0,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_only_last_epoch = save_only_last_epoch
        self.skip_initial_epochs_validation = skip_initial_epochs_validation

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.model(*args, **kwargs)

    def _step(self, batch: Any) -> Any:
        with torch.no_grad():
            input_ids, attention_mask, labels = trim_batch(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            )
        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: Any, _batch_idx: int) -> Any:  # type: ignore
        output = self._step(batch)
        loss = output.loss
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
            batch_size=len(batch["input_ids"]),
        )
        return loss

    def validation_step(self, batch: Any, _batch_idx: int) -> Any:  # type: ignore
        output = self._step(batch)
        loss = output.loss
        if self.current_epoch < self.skip_initial_epochs_validation:
            return {"loss": loss}
        metrics = evaluate_batch(self.model, self.tokenizer, batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
            batch_size=len(batch["input_ids"]),
        )
        return {"loss": loss, "metrics": metrics}

    def test_step(self, batch: Any, _batch_idx: int) -> Any:  # type: ignore
        output = self._step(batch)
        loss = output.loss
        metrics = evaluate_batch(self.model, self.tokenizer, batch)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
            batch_size=len(batch["input_ids"]),
        )
        return {"loss": loss, "metrics": metrics}

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.lr)

    def training_epoch_end(self, training_step_outputs: list[Any]) -> None:
        """save tokenizer and model on epoch end"""
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.output_dir}/epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        if (
            not self.save_only_last_epoch
            or self.current_epoch == self.trainer.max_epochs - 1
        ):
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs: list[Any]) -> None:
        losses = [out["loss"].cpu() for out in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(losses)).item(),
            4,
        )
        if self.current_epoch < self.skip_initial_epochs_validation:
            # no validation metrics to calculate in this epoch, just return early
            return

        metrics = merge_metrics([out["metrics"] for out in validation_step_outputs])
        for task_name, counts in metrics.items():
            self.log(f"val_{task_name}_f1", calc_eval_metrics(*counts)["f_score"])

    def test_epoch_end(self, test_step_outputs: list[Any]) -> None:
        metrics = merge_metrics([out["metrics"] for out in test_step_outputs])
        for task_name, counts in metrics.items():
            self.log(f"test_{task_name}_f1", calc_eval_metrics(*counts)["f_score"])


def merge_metrics(metrics: list[dict[str, list[int]]]) -> dict[str, list[int]]:
    merged_metrics: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for metric in metrics:
        for task_name, counts in metric.items():
            merged_metrics[task_name][0] += counts[0]
            merged_metrics[task_name][1] += counts[1]
            merged_metrics[task_name][2] += counts[2]
    return merged_metrics


def train(
    base_model: str = "t5-base",
    batch_size: int = 8,
    max_epochs: int = 10,
    use_gpu: bool = torch.cuda.is_available(),
    output_dir: str = "outputs",
    early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
    precision: int = 32,
    lr: float = 1e-4,
    num_workers: int = DEFAULT_NUM_WORKERS,
    save_only_last_epoch: bool = False,
    balance_tasks: bool = True,
    max_task_duplication_factor: int = 2,
    skip_initial_epochs_validation: int = 0,
) -> tuple[T5ForConditionalGeneration, T5Tokenizer]:
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info("loading base T5 model")
    model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)
    tokenizer = T5Tokenizer.from_pretrained(
        base_model, model_max_length=MODEL_MAX_LENGTH
    )

    logger.info("loading train/test/val datasets")
    train_dataset = TaskSampleDataset(
        load_sesame_train_samples(),
        tokenizer,
        balance_tasks=balance_tasks,
        max_task_duplication_factor=max_task_duplication_factor,
        augment_data=True,
    )
    val_dataset = TaskSampleDataset(
        load_sesame_dev_samples(),
        tokenizer,
        balance_tasks=False,
    )
    test_dataset = TaskSampleDataset(
        load_sesame_test_samples(),
        tokenizer,
        balance_tasks=False,
    )

    data_module = TrainDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_wrapper = TrainingModelWrapper(
        model,
        tokenizer,
        lr=lr,
        output_dir=output_dir,
        save_only_last_epoch=save_only_last_epoch,
        skip_initial_epochs_validation=skip_initial_epochs_validation,
    )

    # add callbacks
    callbacks: list[Callback] = [TQDMProgressBar(refresh_rate=5)]

    if early_stopping_patience_epochs > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=early_stopping_patience_epochs,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    # prepare trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=1 if use_gpu else 0,
        precision=precision,
        log_every_n_steps=1,
    )

    logger.info("beginning training")

    # fit trainer
    trainer.fit(model_wrapper, data_module)

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a new frame semantic transformer model using FrameNet 1.7"
    )
    parser.add_argument(
        "--base-model",
        default="t5-base",
        help="The HuggingFace T5 model to use as a starting point, default t5-base",
    )
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="default 8")
    parser.add_argument("--epochs", type=int, default=10, help="default 10")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="dir where output models will be saved after each epoch, default ./outputs",
    )
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    train(
        base_model=args.base_model,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        use_gpu=args.use_gpu,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
    )

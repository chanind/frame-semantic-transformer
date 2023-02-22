from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.data_utils import trim_batch
from frame_semantic_transformer.training.evaluate_batch import (
    calc_eval_metrics,
    evaluate_batch,
)


class TrainingModelWrapper(pl.LightningModule):
    """
    Based on https://github.com/Shivanandroy/simpleT5/blob/main/simplet5/simplet5.py
    """

    lr: float
    model: T5ForConditionalGeneration
    tokenizer: T5TokenizerFast
    trainer: pl.Trainer
    output_dir: str
    save_only_last_epoch: bool
    skip_initial_epochs_validation: int
    loader_cache: LoaderDataCache

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        loader_cache: LoaderDataCache,
        lr: float = 1e-4,
        output_dir: str = "outputs",
        save_only_last_epoch: bool = False,
        skip_initial_epochs_validation: int = 0,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.loader_cache = loader_cache
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
        metrics = evaluate_batch(self.model, self.tokenizer, batch, self.loader_cache)
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
        metrics = evaluate_batch(self.model, self.tokenizer, batch, self.loader_cache)
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
            or self.current_epoch == (self.trainer.max_epochs or 0) - 1
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

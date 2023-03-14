from __future__ import annotations
from collections import defaultdict
import json
from os import path, makedirs
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import trim_batch
from frame_semantic_transformer.training.ModelRecorder import ModelRecorder
from .evaluate_batch import TaskEvalResults, calc_eval_metrics, evaluate_batch


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
    val_metrics: dict[str, float] | None
    lr_gamma: float
    log_eval_failures: bool
    model_recorder: ModelRecorder

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        loader_cache: LoaderDataCache,
        lr: float = 1e-4,
        output_dir: str = "outputs",
        save_only_last_epoch: bool = False,
        skip_initial_epochs_validation: int = 0,
        lr_gamma: float = 1.0,
        log_eval_failures: bool = False,
        remove_non_optimal_models: bool = True,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.loader_cache = loader_cache
        self.output_dir = output_dir
        self.save_only_last_epoch = save_only_last_epoch
        self.skip_initial_epochs_validation = skip_initial_epochs_validation
        self.val_metrics = None
        self.lr_gamma = lr_gamma
        self.log_eval_failures = log_eval_failures
        self.model_recorder = ModelRecorder(output_dir)
        self.remove_non_optimal_models = remove_non_optimal_models

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

    def configure_optimizers(self) -> tuple[list[AdamW], list[ExponentialLR]]:
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma, verbose=True)
        return [optimizer], [scheduler]

    def training_epoch_end(self, training_step_outputs: list[Any]) -> None:
        """save tokenizer and model on epoch end"""
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        self.log("train_loss", self.average_training_loss)
        if (
            not self.save_only_last_epoch
            or self.current_epoch == (self.trainer.max_epochs or 0) - 1
        ):
            self.model_recorder.save_model(
                self.model,
                self.tokenizer,
                epoch=self.current_epoch,
                val_loss=self.average_validation_loss,
                task_val_metrics=self.val_metrics,
            )
            if self.remove_non_optimal_models:
                self.model_recorder.remove_non_optimal_models()

    def validation_epoch_end(self, validation_step_outputs: list[Any]) -> None:
        losses = [out["loss"].cpu() for out in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(losses)).item(),
            4,
        )
        self.log("val_loss", self.average_validation_loss)
        if self.current_epoch < self.skip_initial_epochs_validation:
            # no validation metrics to calculate in this epoch, just return early
            return

        metrics = merge_metrics([out["metrics"] for out in validation_step_outputs])
        self.val_metrics = {}
        for task_name, results in metrics.items():
            scores = calc_eval_metrics(results.scores)
            name = f"val_{task_name}_f1"
            f_score = scores["f_score"]
            self.val_metrics[name] = f_score
            self.log(name, f_score)
            self.log(f"val_{task_name}_recall", scores["recall"])
            self.log(f"val_{task_name}_precision", scores["precision"])

        if self.log_eval_failures:
            log_eval_failures(
                self.output_dir,
                f"val_{self.current_epoch}_eval_failures.json",
                metrics,
            )

    def test_epoch_end(self, test_step_outputs: list[Any]) -> None:
        average_test_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in test_step_outputs])).item(),
            4,
        )
        self.log("test_loss", average_test_loss)
        metrics = merge_metrics([out["metrics"] for out in test_step_outputs])
        for task_name, results in metrics.items():
            scores = calc_eval_metrics(results.scores)
            self.log(f"test_{task_name}_f1", scores["f_score"])
            self.log(f"test_{task_name}_recall", scores["recall"])
            self.log(f"test_{task_name}_precision", scores["precision"])
        if self.log_eval_failures:
            log_eval_failures(
                self.output_dir,
                f"test_{self.current_epoch}_eval_failures.json",
                metrics,
            )


def merge_metrics(
    metrics: list[dict[str, TaskEvalResults]]
) -> dict[str, TaskEvalResults]:
    merged_metrics: dict[str, TaskEvalResults] = defaultdict(TaskEvalResults)
    for metric in metrics:
        for task_name, eval_results in metric.items():
            merged_metrics[task_name].scores.false_pos += eval_results.scores.false_pos
            merged_metrics[task_name].scores.false_neg += eval_results.scores.false_neg
            merged_metrics[task_name].scores.true_pos += eval_results.scores.true_pos
            merged_metrics[task_name].false_negatives += eval_results.false_negatives
            merged_metrics[task_name].false_positives += eval_results.false_positives
    return merged_metrics


def log_eval_failures(
    output_dir: str, filename: str, eval_results: dict[str, TaskEvalResults]
) -> None:
    serialized_results = {
        task: results.serialize() for task, results in eval_results.items()
    }
    failures_file = path.join(output_dir, filename)
    makedirs(output_dir, exist_ok=True)
    with open(failures_file, "w+") as f:
        json.dump(serialized_results, f, indent=2)

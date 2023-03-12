from __future__ import annotations
from dataclasses import dataclass
import shutil
from typing import Optional

from transformers import T5ForConditionalGeneration, T5TokenizerFast


class ModelRecorder:

    output_dir: str
    records: list[ModelSaveRecord]

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.records = []

    def save_model(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        epoch: int,
        val_loss: float,
        task_val_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        save_path = self.get_save_path(epoch, val_loss, task_val_metrics)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        self.records.append(
            ModelSaveRecord(
                epoch=epoch,
                val_loss=val_loss,
                task_val_metrics=task_val_metrics,
                save_path=save_path,
            )
        )

    def remove_non_optimal_models(self) -> None:
        """
        Delete any saved model that doesn't have the best validation loss or
        best validation metric for any task.
        """
        best_val_loss_model = _find_best_val_loss_model(self.records)
        best_val_metric_models = _find_best_val_metric_models(self.records)
        optimal_models = [best_val_loss_model, *best_val_metric_models.values()]
        # clone the list while iterating so we can remove items in the loop
        for record in [*self.records]:
            if record not in optimal_models:
                shutil.rmtree(record.save_path)
                self.records.remove(record)

    def get_save_path(
        self,
        epoch: int,
        val_loss: float,
        task_val_metrics: Optional[dict[str, float]] = None,
    ) -> str:
        filename_parts = [
            f"epoch={epoch}",
            f"val_loss={val_loss}",
        ]
        if task_val_metrics:
            filename_parts.extend(
                [f"{k.replace('-', '_')}={v}" for k, v in task_val_metrics.items()]
            )

        return f"{self.output_dir}/{'-'.join(filename_parts)}"


# moved outside of class for easier testing
def _find_best_val_metric_models(
    records: list[ModelSaveRecord],
) -> dict[str, ModelSaveRecord]:
    best_val_metric_models: dict[str, ModelSaveRecord] = {}
    for record in records:
        if record.task_val_metrics:
            for task, val_metric in record.task_val_metrics.items():
                best_val_model = best_val_metric_models.get(task)
                best_val_metrics = (
                    best_val_model.task_val_metrics if best_val_model else {}
                ) or {}
                best_task_metric = best_val_metrics.get(task)
                if best_task_metric is None or val_metric > best_task_metric:
                    best_val_metric_models[task] = record
    return best_val_metric_models


# moved outside of class for easier testing
def _find_best_val_loss_model(records: list[ModelSaveRecord]) -> ModelSaveRecord:
    best_val_loss_model = records[0]
    for record in records[1:]:
        if record.val_loss < best_val_loss_model.val_loss:
            best_val_loss_model = record
    return best_val_loss_model


@dataclass
class ModelSaveRecord:
    epoch: int
    val_loss: float
    task_val_metrics: Optional[dict[str, float]]
    save_path: str

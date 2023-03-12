from __future__ import annotations
from unittest.mock import MagicMock

from frame_semantic_transformer.training.ModelRecorder import (
    ModelRecorder,
    ModelSaveRecord,
    _find_best_val_loss_model,
    _find_best_val_metric_models,
)


def test_ModelRecorder_get_save_path() -> None:
    recorder = ModelRecorder("output_dir")
    save_path = recorder.get_save_path(1, 0.5, {"task1": 0.5, "task2": 0.6})
    assert save_path == "output_dir/epoch=1-val_loss=0.5-task1=0.5-task2=0.6"


def test_ModelRecorder_get_save_path_no_task_val_metrics() -> None:
    recorder = ModelRecorder("output_dir")
    save_path = recorder.get_save_path(1, 0.5)
    assert save_path == "output_dir/epoch=1-val_loss=0.5"


def test_ModelRecorder_get_save_path_replaces_dashes_with_underscores_in_tasks() -> None:
    recorder = ModelRecorder("output_dir")
    save_path = recorder.get_save_path(1, 0.5, {"task1-f1": 0.5, "task2-f1": 0.6})
    assert save_path == "output_dir/epoch=1-val_loss=0.5-task1_f1=0.5-task2_f1=0.6"


def test_ModelRecorder_save_model() -> None:
    model = MagicMock()
    tokenizer = MagicMock()
    recorder = ModelRecorder("output_dir")
    recorder.save_model(model, tokenizer, 1, 0.5, {"task1": 0.5, "task2": 0.6})
    model.save_pretrained.assert_called_once_with(
        "output_dir/epoch=1-val_loss=0.5-task1=0.5-task2=0.6"
    )
    tokenizer.save_pretrained.assert_called_once_with(
        "output_dir/epoch=1-val_loss=0.5-task1=0.5-task2=0.6"
    )
    assert recorder.records == [
        ModelSaveRecord(
            epoch=1,
            val_loss=0.5,
            task_val_metrics={"task1": 0.5, "task2": 0.6},
            save_path="output_dir/epoch=1-val_loss=0.5-task1=0.5-task2=0.6",
        )
    ]


def test_find_best_val_metric_models() -> None:
    epoch1 = ModelSaveRecord(
        epoch=1,
        val_loss=0.5,
        task_val_metrics=None,
        save_path="...",
    )
    epoch2 = ModelSaveRecord(
        epoch=2,
        val_loss=0.4,
        task_val_metrics={"task1": 0.6, "task2": 0.7, "task3": 0.8},
        save_path="...",
    )
    epoch3 = ModelSaveRecord(
        epoch=3,
        val_loss=0.3,
        task_val_metrics={"task1": 0.7, "task2": 0.8, "task3": 0.7},
        save_path="...",
    )
    model_records = [epoch1, epoch2, epoch3]
    assert _find_best_val_metric_models(model_records) == {
        "task1": epoch3,
        "task2": epoch3,
        "task3": epoch2,
    }


def test_find_best_val_loss_model() -> None:
    epoch1 = ModelSaveRecord(
        epoch=1,
        val_loss=0.43,
        task_val_metrics=None,
        save_path="...",
    )
    epoch2 = ModelSaveRecord(
        epoch=2,
        val_loss=0.4,
        task_val_metrics=None,
        save_path="...",
    )
    epoch3 = ModelSaveRecord(
        epoch=3,
        val_loss=0.45,
        task_val_metrics={"task1": 0.7, "task2": 0.8, "task3": 0.7},
        save_path="...",
    )
    model_records = [epoch1, epoch2, epoch3]
    assert _find_best_val_loss_model(model_records) == epoch2

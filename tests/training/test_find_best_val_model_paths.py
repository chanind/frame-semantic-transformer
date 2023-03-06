from __future__ import annotations

from frame_semantic_transformer.training.find_best_val_model_paths import (
    get_model_scores,
)


def test_get_model_scores() -> None:
    model_path = "epoch=1--val_loss=1.0000--val_args_extraction_f1=0.5000--val_trigger_identification_f1=0.4000--val_frame_classification_f1=0.6000"
    expected_scores = {
        "val_loss": 1.0,
        "val_args_extraction_f1": 0.5,
        "val_trigger_identification_f1": 0.4,
        "val_frame_classification_f1": 0.6,
        "val_avg_f1": 0.5,
    }
    assert get_model_scores(model_path) == expected_scores

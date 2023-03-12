from __future__ import annotations

import glob
import os


KEYS_TO_CHECK = {
    "val_loss": {"minimize": True},
    "val_args_extraction_f1": {"minimize": False},
    "val_trigger_identification_f1": {"minimize": False},
    "val_frame_classification_f1": {"minimize": False},
    "val_avg_f1": {"minimize": False},
}


def find_best_val_model_paths(outputs_dir: str) -> dict[str, str]:
    """
    Helper script to find the models with the higest validation scores by f1 or val loss after training
    """
    potential_outputs = [os.path.basename(obj) for obj in glob.glob(f"{outputs_dir}/*")]

    top_scores: dict[str, float] = {}
    best_models = {}
    for output_name in potential_outputs:
        if "epoch=" not in output_name:
            continue
        scores = get_model_scores(output_name)
        for key, value in scores.items():
            if (
                key not in top_scores
                or (KEYS_TO_CHECK[key]["minimize"] and value < top_scores[key])
                or (not KEYS_TO_CHECK[key]["minimize"] and value > top_scores[key])
            ):
                top_scores[key] = value
                best_models[key] = output_name
    return best_models


def get_model_scores(output_name: str) -> dict[str, float]:
    """
    Helper function to get the scores for a given model
    """
    scores = {}
    for name_part in output_name.split("-"):
        if "=" in name_part:
            key, value = name_part.split("=")
            if key in KEYS_TO_CHECK:
                scores[key] = float(value)
    if (
        "val_args_extraction_f1" in scores
        and "val_trigger_identification_f1" in scores
        and "val_frame_classification_f1" in scores
    ):
        scores["val_avg_f1"] = (
            scores["val_args_extraction_f1"]
            + scores["val_trigger_identification_f1"]
            + scores["val_frame_classification_f1"]
        ) / 3
    return scores

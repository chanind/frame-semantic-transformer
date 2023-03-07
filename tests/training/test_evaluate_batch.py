from __future__ import annotations

from frame_semantic_transformer.training.evaluate_batch import (
    TaskEvalResults,
    EvalFailure,
    EvalScores,
)


def test_serialize_TaskEvalResults() -> None:
    results = TaskEvalResults(
        scores=EvalScores(
            true_pos=10,
            false_pos=5,
            false_neg=2,
        ),
        false_positives=[
            EvalFailure("input", "target", ["pred1", "pred2"]),
        ],
        false_negatives=[
            EvalFailure("input", "target", ["pred1", "pred2"]),
        ],
    )
    assert results.serialize() == {
        "scores": {
            "true_pos": 10,
            "false_pos": 5,
            "false_neg": 2,
        },
        "false_positives": [
            {
                "input": "input",
                "target": "target",
                "predictions": ["pred1", "pred2"],
            },
        ],
        "false_negatives": [
            {
                "input": "input",
                "target": "target",
                "predictions": ["pred1", "pred2"],
            },
        ],
    }

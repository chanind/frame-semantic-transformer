from __future__ import annotations

from frame_semantic_transformer.data.tasks.FrameClassificationSample import (
    FrameClassificationSample,
)
from frame_semantic_transformer.data.tasks.FrameClassificationTask import (
    FrameClassificationTask,
)


sample = FrameClassificationSample(
    task=FrameClassificationTask(
        text="Your contribution to Goodwill will mean more than you may know .",
        trigger_loc=5,
    ),
    frame="Giving",
)


def test_get_input() -> None:
    expected = "FRAME Giving : Your * contribution to Goodwill will mean more than you may know."
    assert sample.get_input() == expected


def test_get_target() -> None:
    expected = "Giving"
    assert sample.get_target() == expected


def test_evaluate_prediction_correct_prediction() -> None:
    correct_pred = "Giving"
    assert FrameClassificationSample.evaluate_prediction(
        [correct_pred],
        correct_pred,
        "FRAME: Your * contribution to Goodwill will mean more than you may know.",
    ) == (1, 0, 0)


def test_evaluate_prediction_increments_fp_and_fn_on_incorrect_pred() -> None:
    incorrect_pred = "Aiming"
    nonsense_pred = "Nonsense"
    assert FrameClassificationSample.evaluate_prediction(
        [incorrect_pred],
        "Giving",
        "FRAME: Your * contribution to Goodwill will mean more than you may know.",
    ) == (
        0,
        1,
        1,
    )
    assert FrameClassificationSample.evaluate_prediction(
        [nonsense_pred],
        "Giving",
        "FRAME: Your * contribution to Goodwill will mean more than you may know.",
    ) == (
        0,
        1,
        1,
    )

from __future__ import annotations

import pytest

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.tasks.FrameClassificationSample import (
    FrameClassificationSample,
)
from frame_semantic_transformer.data.tasks.FrameClassificationTask import (
    FrameClassificationTask,
)


@pytest.fixture
def sample(loader_cache: LoaderDataCache) -> FrameClassificationSample:
    return FrameClassificationSample(
        task=FrameClassificationTask(
            text="Your contribution to Goodwill will mean more than you may know .",
            trigger_loc=5,
            loader_cache=loader_cache,
        ),
        frame="Giving",
    )


def test_get_input(sample: FrameClassificationSample) -> None:
    expected = "FRAME Condition_symptom_relation Giving : Your * contribution to Goodwill will mean more than you may know."
    assert sample.get_input() == expected


def test_get_target(sample: FrameClassificationSample) -> None:
    expected = "Giving"
    assert sample.get_target() == expected


def test_evaluate_prediction_correct_prediction(loader_cache: LoaderDataCache) -> None:
    correct_pred = "Giving"
    assert FrameClassificationSample.evaluate_prediction(
        [correct_pred],
        correct_pred,
        "FRAME: Your * contribution to Goodwill will mean more than you may know.",
        loader_cache,
    ) == (1, 0, 0)


def test_evaluate_prediction_increments_fp_and_fn_on_incorrect_pred(
    loader_cache: LoaderDataCache,
) -> None:
    incorrect_pred = "Aiming"
    nonsense_pred = "Nonsense"
    assert FrameClassificationSample.evaluate_prediction(
        [incorrect_pred],
        "Giving",
        "FRAME: Your * contribution to Goodwill will mean more than you may know.",
        loader_cache,
    ) == (
        0,
        1,
        1,
    )
    assert FrameClassificationSample.evaluate_prediction(
        [nonsense_pred],
        "Giving",
        "FRAME: Your * contribution to Goodwill will mean more than you may know.",
        loader_cache,
    ) == (
        0,
        1,
        1,
    )

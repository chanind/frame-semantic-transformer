from __future__ import annotations

import pytest

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.frame_types import FrameElementAnnotation

from frame_semantic_transformer.data.tasks.ArgumentsExtractionSample import (
    ArgumentsExtractionSample,
)
from frame_semantic_transformer.data.tasks.ArgumentsExtractionTask import (
    ArgumentsExtractionTask,
)


@pytest.fixture
def sample(loader_cache: LoaderDataCache) -> ArgumentsExtractionSample:
    return ArgumentsExtractionSample(
        task=ArgumentsExtractionTask(
            text="Your contribution to Goodwill will mean more than you may know .",
            trigger_loc=5,
            frame="Giving",
            loader_cache=loader_cache,
        ),
        frame_elements=[
            FrameElementAnnotation(
                name="Donor",
                start_loc=0,
                end_loc=4,
            ),
            FrameElementAnnotation(
                name="Recipient",
                start_loc=18,
                end_loc=29,
            ),
        ],
    )


def test_get_input(sample: ArgumentsExtractionSample) -> None:
    elements = "Donor Recipient Theme Place Explanation Time Purpose Means Manner Circumstances Imposed_purpose Depictive Period_of_iterations"
    expected = f"ARGS Giving | {elements} : Your * contribution to Goodwill will mean more than you may know."
    assert sample.get_input() == expected


def test_get_target(sample: ArgumentsExtractionSample) -> None:
    expected = "Donor = Your | Recipient = to Goodwill"
    assert sample.get_target() == expected


def test_evaluate_prediction_just_does_a_simple_string_match_for_now(
    sample: ArgumentsExtractionSample,
    loader_cache: LoaderDataCache,
) -> None:
    target = "Donor = Your | Recipient = to Goodwill"
    incorrect_pred = "Donor = Your | Recipient = Me | Bleh = so what"
    assert ArgumentsExtractionSample.evaluate_prediction(
        [target], target, sample.get_input(), loader_cache
    ) == (
        2.0,
        0.0,
        0.0,
    )
    assert ArgumentsExtractionSample.evaluate_prediction(
        [incorrect_pred], target, sample.get_input(), loader_cache
    ) == (
        1.0,
        2.0,
        1.0,
    )

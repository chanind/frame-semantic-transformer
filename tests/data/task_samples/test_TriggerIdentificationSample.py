from __future__ import annotations

from frame_semantic_transformer.data.task_samples.TriggerIdentificationSample import (
    TriggerIdentificationSample,
)


sample = TriggerIdentificationSample(
    text="Your contribution to Goodwill will mean more than you may know .",
    trigger_locs=[5, 18, 35, 40, 58, 54],
)


def test_get_input() -> None:
    expected = (
        "TRIGGER: Your contribution to Goodwill will mean more than you may know ."
    )
    assert sample.get_input() == expected


def test_get_target() -> None:
    expected = "Your *contribution *to Goodwill will *mean *more than you *may *know ."
    assert sample.get_target() == expected


def test_evaluate_prediction() -> None:
    pred = "Your contribution *to Goodwill *will *mean *more than you may *know ."
    assert sample.evaluate_prediction(pred) == (4, 1, 2)


def test_evaluate_prediction_fails_for_elements_whose_content_doesnt_match() -> None:
    pred = "Your AHAHAHAHA *to BADWILL will *PSYCH *more than you may *know ."
    assert sample.evaluate_prediction(pred) == (3, 1, 3)


def test_evaluate_prediction_treats_missing_words_as_wrong() -> None:
    pred = "Your *contribution *to Goodwill will *mean"
    assert sample.evaluate_prediction(pred) == (3, 3, 3)


def test_evaluate_prediction_treats_excess_words_as_false_positives() -> None:
    pred = "Your *contribution *to Goodwill will *mean *more than you *may *know . ha ha ha ha!"
    assert sample.evaluate_prediction(pred) == (6, 4, 0)

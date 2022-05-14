from __future__ import annotations

from frame_semantic_transformer.data.task_samples.ArgumentsExtractionSample import (
    ArgumentsExtractionSample,
)


sample = ArgumentsExtractionSample(
    text="Your contribution to Goodwill will mean more than you may know .",
    trigger_loc=(5, 17),
    frame="Giving",
    frame_element_locs=[(0, 4, "Donor"), (18, 29, "Recipient")],
)


def test_get_input() -> None:
    elements = "Donor Recipient Theme Place Explanation Time Purpose Means Manner Circumstances Imposed_purpose Depictive Period_of_iterations"
    expected = f"ARGS Giving | {elements} : Your * contribution * to Goodwill will mean more than you may know ."
    assert sample.get_input() == expected


def test_get_target() -> None:
    expected = "Donor = Your | Recipient = to Goodwill"
    assert sample.get_target() == expected


def test_evaluate_prediction_just_does_a_simple_string_match_for_now() -> None:
    target = "Donor = Your | Recipient = to Goodwill"
    incorrect_pred = "Donor = Your | Recipient = to Goodwill | meh"
    assert ArgumentsExtractionSample.evaluate_prediction([target], target) == (1, 0, 0)
    assert ArgumentsExtractionSample.evaluate_prediction([incorrect_pred], target) == (
        0,
        1,
        0,
    )

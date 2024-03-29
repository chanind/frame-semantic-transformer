from __future__ import annotations
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.tasks.TriggerIdentificationSample import (
    TriggerIdentificationSample,
    process_text_for_evaluation,
)
from frame_semantic_transformer.data.tasks.TriggerIdentificationTask import (
    TriggerIdentificationTask,
)

sample = TriggerIdentificationSample(
    task=TriggerIdentificationTask(
        text="Your contribution to Goodwill will mean more than you may know.",
    ),
    trigger_locs=[5, 18, 35, 40, 58, 54],
)

target = "Your * contribution * to Goodwill will * mean * more than you * may * know."


def test_get_input() -> None:
    expected = (
        "TRIGGER: Your contribution to Goodwill will mean more than you may know."
    )
    assert sample.get_input() == expected


def test_get_target() -> None:
    expected = (
        "Your * contribution * to Goodwill will * mean * more than you * may * know."
    )
    assert sample.get_target() == expected


def test_evaluate_prediction(loader_cache: LoaderDataCache) -> None:
    pred = "Your contribution * to Goodwill * will * mean * more than you may * know."
    assert TriggerIdentificationSample.evaluate_prediction(
        [pred], target, sample.get_input(), loader_cache
    ) == (4, 1, 2)


def test_evaluate_prediction_fails_for_elements_whose_content_doesnt_match(
    loader_cache: LoaderDataCache,
) -> None:
    pred = "Your AHAHAHAHA * to BADWILL will * PSYCH * more than you may * know."
    assert TriggerIdentificationSample.evaluate_prediction(
        [pred], target, sample.get_input(), loader_cache
    ) == (3, 1, 3)


def test_evaluate_prediction_treats_missing_words_as_wrong(
    loader_cache: LoaderDataCache,
) -> None:
    pred = "Your * contribution * to Goodwill will * mean"
    assert TriggerIdentificationSample.evaluate_prediction(
        [pred], target, sample.get_input(), loader_cache
    ) == (3, 2, 3)


def test_evaluate_prediction_treats_excess_words_as_false_positives(
    loader_cache: LoaderDataCache,
) -> None:
    pred = "Your * contribution * to Goodwill will * mean * more than you * may * know. ha ha ha ha!"
    assert TriggerIdentificationSample.evaluate_prediction(
        [pred], target, sample.get_input(), loader_cache
    ) == (6, 4, 0)


def test_process_text_for_evaluation_handles_contractions() -> None:
    assert process_text_for_evaluation("couldn't") == "couldn 't"
    assert process_text_for_evaluation("couldn*'t") == "couldn *'t"

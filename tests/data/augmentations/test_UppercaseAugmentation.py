from __future__ import annotations
from typing import cast
import pytest

from frame_semantic_transformer.data.augmentations import UppercaseAugmentation
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[16],
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        ("I am a banana.", "I AM A BANANA."),
        ("I AM A banana !", "I AM A BANANA !"),
    ],
)
def test_UppercaseAugmentation(input: str, expected: str) -> None:
    augmentation = UppercaseAugmentation(1.0)
    sample = create_trigger_identification_sample(input)
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == expected


def test_UppercaseAugmentation_returns_original_sentence_if_contains_ligature() -> None:
    sentence = "this is one character: ﬁ ."
    augmentation = UppercaseAugmentation(1.0)
    sample = create_trigger_identification_sample(sentence)
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == sentence

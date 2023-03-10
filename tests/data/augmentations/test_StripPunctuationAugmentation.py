from __future__ import annotations
from typing import cast

from frame_semantic_transformer.data.augmentations import StripPunctuationAugmentation
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[0],
    )


def test_StripPunctuationAugmentation_removes_punctuation() -> None:
    augmentation = StripPunctuationAugmentation(1.0, min_to_remove=5, max_to_remove=5)
    sample = create_trigger_identification_sample("This! is? A! sentence.")
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == "This is A sentence"


def test_StripPunctuationAugmentation_removes_up_to_max_to_remove() -> None:
    augmentation = StripPunctuationAugmentation(1.0, min_to_remove=5, max_to_remove=5)
    sample = create_trigger_identification_sample("This! is! A! sentence !!! !")
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text.count("!") == 2

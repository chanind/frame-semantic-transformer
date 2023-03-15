from __future__ import annotations
from typing import cast

from frame_semantic_transformer.data.augmentations import SimpleMisspellingAugmentation
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[16],
    )


def test_SimpleMisspellingAugmentation() -> None:
    sentence = "I like to eat food 1234"
    # just to make it almost certain something will be changed
    augmentation = SimpleMisspellingAugmentation(
        1.0, min_misspellings_per_sentence=20, max_misspellings_per_sentence=20
    )
    sample = create_trigger_identification_sample(sentence)
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    new_sentence = new_sample.task.text
    assert new_sentence != sentence
    assert len(new_sentence) == len(sentence)
    assert len(new_sentence.split()) == len(sentence.split())

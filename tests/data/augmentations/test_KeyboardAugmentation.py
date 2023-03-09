from __future__ import annotations
from typing import cast

from frame_semantic_transformer.data.augmentations import KeyboardAugmentation
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[16],
    )


def test_KeyboardAugmentation() -> None:
    sentence = "I like to eat food 1234 and I like in a boat"
    # just to make it almost certain something will be changed
    augmentation = KeyboardAugmentation(1.0)
    sample = create_trigger_identification_sample(sentence)

    is_same = True
    # do this 20 times since it's not guaranteed to change anything every time
    for _ in range(20):
        new_sample = cast(TriggerIdentificationSample, augmentation(sample))
        new_sentence = new_sample.task.text
        if new_sentence != sentence:
            is_same = False
        assert len(new_sentence) == len(sentence)
        assert len(new_sentence.split()) == len(sentence.split())
    assert not is_same

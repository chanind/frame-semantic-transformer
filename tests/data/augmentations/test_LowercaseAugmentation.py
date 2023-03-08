from __future__ import annotations
from typing import cast
import pytest

from frame_semantic_transformer.data.augmentations import LowercaseAugmentation
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
        ("I am a banana.", "i am a banana."),
        ("I AM A BANANA !", "i am a banana !"),
    ],
)
def test_LowercaseAugmentation(input: str, expected: str) -> None:
    augmentation = LowercaseAugmentation(1.0)
    sample = create_trigger_identification_sample(input)
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == expected

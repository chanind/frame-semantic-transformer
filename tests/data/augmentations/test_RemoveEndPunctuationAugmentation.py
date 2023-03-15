from __future__ import annotations
from typing import cast
import pytest

from frame_semantic_transformer.data.augmentations import (
    RemoveEndPunctuationAugmentation,
)
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[2],
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        ("I am a banana.", "I am a banana"),
        ("I am a banana!", "I am a banana"),
        ("I ! am a banana .", "I ! am a banana"),
    ],
)
def test_RemoveEndPunctuationAugmentation(input: str, expected: str) -> None:
    augmentation = RemoveEndPunctuationAugmentation(1.0)
    sample = create_trigger_identification_sample(input)
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == expected

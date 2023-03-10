from __future__ import annotations
from typing import cast
import pytest

from frame_semantic_transformer.data.augmentations import DoubleQuotesAugmentation
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[0],
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        ("I am a ``banana'' .", 'I am a "banana" .'),
        ("she ``says'' ``hi''", 'she "says" "hi"'),
    ],
)
def test_DoubleQuotesAugmentation_changes_latex_quotes_to_standard_quotes(
    input: str, expected: str
) -> None:
    augmentation = DoubleQuotesAugmentation(1.0)
    sample = create_trigger_identification_sample(input)
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == expected


def test_DoubleQuotesAugmentation_changes_standard_quotes_to_latex_quotes() -> None:
    augmentation = DoubleQuotesAugmentation(1.0)
    sample = create_trigger_identification_sample('This is a quote: " .')
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text in {"This is a quote: '' .", "This is a quote: `` ."}


def test_DoubleQuotesAugmentation_leaves_samples_unchanged_if_no_quotes_are_present() -> None:
    augmentation = DoubleQuotesAugmentation(1.0)
    sample = create_trigger_identification_sample("Nothing to see here !")
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == sample.task.text

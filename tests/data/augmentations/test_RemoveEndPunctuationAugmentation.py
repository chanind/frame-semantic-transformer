from __future__ import annotations
import pytest

from frame_semantic_transformer.data.augmentations import (
    RemoveEndPunctuationAugmentation,
)


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            ("TASK: I am a banana.", "I am a banana."),
            ("TASK: I am a banana", "I am a banana"),
        ),
        (
            ("TASK: I am a banana!", "I am a banana!"),
            ("TASK: I am a banana", "I am a banana"),
        ),
        (
            ("TASK: I am a banana .", "I am a banana ."),
            ("TASK: I am a banana", "I am a banana"),
        ),
    ],
)
def test_RemoveEndPunctuationAugmentation(
    input: tuple[str, str], expected: tuple[str, str]
) -> None:
    augmentation = RemoveEndPunctuationAugmentation(1.0)
    assert augmentation(*input) == expected

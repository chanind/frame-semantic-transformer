from __future__ import annotations
import pytest

from frame_semantic_transformer.data.augmentations import LowercaseAugmentation


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            ("TASK: I am a banana.", "I am a banana."),
            ("TASK: i am a banana.", "i am a banana."),
        ),
        (
            ("TASK: I AM A BANANA !", "I AM A BANANA !"),
            ("TASK: i am a banana !", "i am a banana !"),
        ),
        (
            ("TASK | Param1 | Param 2 : I AM A BANANA !", "I AM A BANANA !"),
            ("TASK | Param1 | Param 2 : i am a banana !", "i am a banana !"),
        ),
        (
            ("TASK: Ch 1: I AM A BANANA !", "Ch 1: I AM A BANANA !"),
            ("TASK: ch 1: i am a banana !", "ch 1: i am a banana !"),
        ),
    ],
)
def test_LowercaseAugmentation(
    input: tuple[str, str], expected: tuple[str, str]
) -> None:
    augmentation = LowercaseAugmentation(1.0)
    assert augmentation(*input) == expected

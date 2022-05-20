from __future__ import annotations
import pytest

from frame_semantic_transformer.data.augmentations.RemoveContractionsAugmentation import (
    RemoveContractionsAugmentation,
)


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            ("TASK: I can't go I won't go", "I can't go I won't go"),
            ("TASK: I cannot go I will not go", "I cannot go I will not go"),
        ),
        (
            (
                "TASK: shouldn't couldn't they're we'll they've",
                "shouldn't couldn't they're we'll they've",
            ),
            (
                "TASK: should not could not they are we will they have",
                "should not could not they are we will they have",
            ),
        ),
        (
            ("TASK | Param1 | Param 2 : We're didn*'t", "We're didn't"),
            ("TASK | Param1 | Param 2 : We're didn*'t", "We're didn't"),
        ),
    ],
)
def test_RemoveContractionsAugmentation(
    input: tuple[str, str], expected: tuple[str, str]
) -> None:
    augmentation = RemoveContractionsAugmentation(1.0)
    assert augmentation(*input) == expected

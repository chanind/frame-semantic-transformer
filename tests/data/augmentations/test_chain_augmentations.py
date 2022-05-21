from __future__ import annotations

from frame_semantic_transformer.data.augmentations import (
    chain_augmentations,
    LowercaseAugmentation,
    RemoveContractionsAugmentation,
    RemoveEndPunctuationAugmentation,
)


def test_chain_augmentations_applys_all_augmentations() -> None:
    augmentation = chain_augmentations(
        [
            LowercaseAugmentation(1.0),
            RemoveContractionsAugmentation(1.0),
            RemoveEndPunctuationAugmentation(1.0),
        ]
    )

    input = "TASK: I don't like BANANAS!"
    target = "I don't like BANANAS!"
    assert augmentation(input, target) == (
        "TASK: i do not like bananas",
        "i do not like bananas",
    )

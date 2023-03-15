from __future__ import annotations
from typing import cast

from frame_semantic_transformer.data.augmentations import (
    chain_augmentations,
    LowercaseAugmentation,
    RemoveEndPunctuationAugmentation,
)
from frame_semantic_transformer.data.tasks import (
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[16],
    )


def test_chain_augmentations_applys_all_augmentations() -> None:
    augmentation = chain_augmentations(
        [
            LowercaseAugmentation(1.0),
            RemoveEndPunctuationAugmentation(1.0),
        ]
    )
    sample = create_trigger_identification_sample("I don't like BANANAS!")
    new_sample = cast(TriggerIdentificationSample, augmentation(sample))
    assert new_sample.task.text == "i don't like bananas"

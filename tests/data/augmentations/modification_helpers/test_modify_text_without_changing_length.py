from __future__ import annotations
from typing import cast
from unittest.mock import MagicMock

import pytest

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.augmentations.modification_helpers import (
    modify_text_without_changing_length,
)
from frame_semantic_transformer.data.frame_types import FrameElementAnnotation
from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    ArgumentsExtractionTask,
    FrameClassificationSample,
    FrameClassificationTask,
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)


def create_arg_extraction_sample(
    sentence: str, loader_cache: LoaderDataCache
) -> ArgumentsExtractionSample:
    return ArgumentsExtractionSample(
        task=ArgumentsExtractionTask(
            text=sentence,
            trigger_loc=15,
            frame="blah",
            loader_cache=loader_cache,
        ),
        frame_elements=[
            FrameElementAnnotation(
                name="The_elm",
                start_loc=4,
                end_loc=9,
            )
        ],
    )


def create_frame_classification_sample(
    sentence: str, loader_cache: LoaderDataCache
) -> FrameClassificationSample:
    return FrameClassificationSample(
        task=FrameClassificationTask(
            text=sentence,
            trigger_loc=15,
            loader_cache=loader_cache,
        ),
        frame="blah",
    )


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[15],
    )


def test_modify_text_without_changing_length_for_arg_extraction_sample(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    new_sentence = "The quick BROWN fox jumps over the LAZY MAN"
    sample = create_arg_extraction_sample(sentence, loader_cache)
    callback = MagicMock(return_value=new_sentence)
    new_sample = cast(
        ArgumentsExtractionSample, modify_text_without_changing_length(sample, callback)
    )
    assert new_sample.task.text == new_sentence
    assert new_sample.task.frame == sample.task.frame
    assert new_sample.task.trigger_loc == sample.task.trigger_loc
    assert new_sample.frame_elements == sample.frame_elements
    callback.assert_called_with(sentence)


def test_throw_error_if_sentence_length_changes(loader_cache: LoaderDataCache) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    new_sentence = "The LOOOOOOOOOONG quick BROWN fox jumps over the LAZY MAN"
    sample = create_arg_extraction_sample(sentence, loader_cache)
    callback = MagicMock(return_value=new_sentence)
    with pytest.raises(ValueError):
        modify_text_without_changing_length(sample, callback)


def test_modify_text_without_changing_length_for_frame_classification_sample(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    new_sentence = "The quick BROWN fox jumps over the LAZY MAN"
    sample = create_frame_classification_sample(sentence, loader_cache)
    callback = MagicMock(return_value=new_sentence)
    new_sample = cast(
        FrameClassificationSample, modify_text_without_changing_length(sample, callback)
    )
    assert new_sample.task.text == new_sentence
    assert new_sample.frame == sample.frame
    assert new_sample.task.trigger_loc == sample.task.trigger_loc
    callback.assert_called_with(sentence)


def test_modify_text_without_changing_length_for_trigger_identification_sample() -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    new_sentence = "The quick BROWN fox jumps over the LAZY MAN"
    sample = create_trigger_identification_sample(sentence)
    callback = MagicMock(return_value=new_sentence)
    new_sample = cast(
        TriggerIdentificationSample,
        modify_text_without_changing_length(sample, callback),
    )
    assert new_sample.task.text == new_sentence
    assert new_sample.trigger_locs == sample.trigger_locs
    callback.assert_called_with(sentence)

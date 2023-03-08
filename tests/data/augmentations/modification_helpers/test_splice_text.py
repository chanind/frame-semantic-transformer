from __future__ import annotations
from typing import cast
from unittest.mock import MagicMock

import pytest

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.augmentations.modification_helpers import (
    splice_text,
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
            trigger_loc=16,
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
            trigger_loc=16,
            loader_cache=loader_cache,
        ),
        frame="blah",
    )


def create_trigger_identification_sample(sentence: str) -> TriggerIdentificationSample:
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=sentence),
        trigger_locs=[16],
    )


def test_splice_text_splices_the_text_into_the_sentence(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    sample = create_arg_extraction_sample(sentence, loader_cache)
    callback = MagicMock(return_value=(20, 10, "EATS"))
    new_sample = cast(ArgumentsExtractionSample, splice_text(sample, callback))
    assert new_sample.task.text == "The quick brown fox EATS the lazy dog"
    assert new_sample.task.frame == sample.task.frame
    assert new_sample.task.trigger_loc == sample.task.trigger_loc
    assert new_sample.frame_elements == sample.frame_elements
    callback.assert_called_with(sentence, [16, 4, 9])


def test_splice_text_leaves_sentence_unchanged_if_callback_returns_none(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    sample = create_arg_extraction_sample(sentence, loader_cache)
    callback = MagicMock(return_value=None)
    new_sample = cast(ArgumentsExtractionSample, splice_text(sample, callback))
    assert new_sample.task.text == sentence
    assert new_sample.task.frame == sample.task.frame
    assert new_sample.task.trigger_loc == sample.task.trigger_loc
    assert new_sample.frame_elements == sample.frame_elements
    callback.assert_called_with(sentence, [16, 4, 9])


def test_splice_text_modified_indices_after_the_changes(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    sample = create_arg_extraction_sample(sentence, loader_cache)
    # replace 'The' with 'The nonexistant'
    callback = MagicMock(return_value=(0, 3, "The nonexistant"))
    new_sample = cast(ArgumentsExtractionSample, splice_text(sample, callback))
    assert (
        new_sample.task.text
        == "The nonexistant quick brown fox jumps over the lazy dog"
    )
    assert new_sample.task.frame == sample.task.frame
    assert new_sample.task.trigger_loc == sample.task.trigger_loc + 12
    assert len(new_sample.frame_elements) == 1
    assert new_sample.frame_elements[0].start_loc == 16
    assert new_sample.frame_elements[0].end_loc == 21
    callback.assert_called_with(sentence, [16, 4, 9])


def test_throw_error_if_splice_text_delete_a_critical_index(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    sample = create_arg_extraction_sample(sentence, loader_cache)
    callback = MagicMock(return_value=(4, 2, ""))
    with pytest.raises(ValueError):
        splice_text(sample, callback)


def test_splice_text_for_frame_classification_sample(
    loader_cache: LoaderDataCache,
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    sample = create_frame_classification_sample(sentence, loader_cache)
    # replace 'The' with 'The nonexistant'
    callback = MagicMock(return_value=(0, 3, "The nonexistant"))
    new_sample = cast(FrameClassificationSample, splice_text(sample, callback))
    assert (
        new_sample.task.text
        == "The nonexistant quick brown fox jumps over the lazy dog"
    )
    assert new_sample.frame == sample.frame
    assert new_sample.task.trigger_loc == sample.task.trigger_loc + 12
    callback.assert_called_with(sentence, [16])


def test_splice_text_for_trigger_identification_sample() -> None:
    sentence = "The quick brown fox jumps over the lazy dog"
    sample = create_trigger_identification_sample(sentence)
    # replace 'The' with 'The nonexistant'
    callback = MagicMock(return_value=(0, 3, "The nonexistant"))
    new_sample = cast(
        TriggerIdentificationSample,
        splice_text(sample, callback),
    )
    assert (
        new_sample.task.text
        == "The nonexistant quick brown fox jumps over the lazy dog"
    )
    assert new_sample.trigger_locs == [28]
    callback.assert_called_with(sentence, [16])

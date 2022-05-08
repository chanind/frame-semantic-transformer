from __future__ import annotations
from typing import Any, Iterable, Mapping, Optional
from frame_semantic_transformer.data.framenet import get_fulltext_docs

from frame_semantic_transformer.data.sesame_data_splits import (
    SESAME_DEV_FILES,
    SESAME_TEST_FILES,
)
from frame_semantic_transformer.data.task_samples.ArgumentsExtractionSample import (
    ArgumentsExtractionSample,
)
from frame_semantic_transformer.data.task_samples.FrameClassificationSample import (
    FrameClassificationSample,
)
from frame_semantic_transformer.data.task_samples.TaskSample import TaskSample
from frame_semantic_transformer.data.task_samples.TriggerIdentificationSample import (
    TriggerIdentificationSample,
)


def load_framenet_samples(
    include_docs: Optional[Iterable[str]] = None,
    exclude_docs: Optional[Iterable[str]] = None,
) -> list[TaskSample]:
    samples: list[TaskSample] = []
    for doc in get_fulltext_docs():
        if exclude_docs and doc["filename"] in exclude_docs:
            continue
        if include_docs and doc["filename"] not in include_docs:
            continue
        samples += parse_samples_from_fulltext_doc(doc)
    return samples


def load_sesame_train_samples() -> list[TaskSample]:
    return load_framenet_samples(exclude_docs=SESAME_DEV_FILES + SESAME_TEST_FILES)


def load_sesame_test_samples() -> list[TaskSample]:
    return load_framenet_samples(include_docs=SESAME_TEST_FILES)


def load_sesame_dev_samples() -> list[TaskSample]:
    return load_framenet_samples(include_docs=SESAME_DEV_FILES)


def extract_annotation_target(targets: list[tuple[int, int]]) -> tuple[int, int] | None:
    """
    For targets like "Ended up", where it's actually just a compount word, combine the targets together
    Else, just return None and we can skip this target
    """
    target = targets[0]
    remaining_targets = targets[1:]
    if len(remaining_targets) == 0:
        return target
    subsequent_target = extract_annotation_target(remaining_targets)
    if not subsequent_target:
        return None
    # until we see a counter-example, assume all multi-word targets are just compound-words with a space between
    target_loc_diff = subsequent_target[0] - target[1]
    if target_loc_diff < 0 or target_loc_diff > 1:
        return None
    return (target[0], subsequent_target[1])


def parse_frame_samples_from_annotation_set(
    annotation_set: Iterable[Mapping[str, Any]]
) -> list[FrameClassificationSample | ArgumentsExtractionSample]:
    """
    Helper to parse sample sentences out of framenet annotation sets
    Not all annotation sets contain frames, so this will filter out any invalid annotation sets
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu.exemplars)
    """
    sample_sentences: list[FrameClassificationSample | ArgumentsExtractionSample] = []
    for annotation in annotation_set:
        if "FE" in annotation and "Target" in annotation and "frame" in annotation:
            if annotation["FE"][1] != {}:
                # I don't understand what the second part of this tuple is, just ignore it for now
                continue
            trigger_loc = extract_annotation_target(annotation["Target"])
            # if the trigger loc is weird for some reason just skip it for now
            if not trigger_loc:
                continue
            sample_sentences.append(
                ArgumentsExtractionSample(
                    text=annotation["text"],
                    trigger_loc=trigger_loc,
                    frame=annotation["frame"]["name"],
                    frame_element_locs=annotation["FE"][0],
                )
            )
            sample_sentences.append(
                FrameClassificationSample(
                    text=annotation["text"],
                    trigger_loc=trigger_loc,
                    frame=annotation["frame"]["name"],
                )
            )
    return sample_sentences


def parse_trigger_sample_from_annotation_set(
    annotation_set: Iterable[Mapping[str, Any]]
) -> TriggerIdentificationSample | None:
    """
    Helper to parse sample sentences out of framenet annotation sets
    Not all annotation sets contain frames, so this will filter out any invalid annotation sets
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu.exemplars)
    """
    target_locs = []
    text: Optional[str] = None
    for annotation in annotation_set:
        if "Target" in annotation:
            text = annotation["text"]
            for loc in annotation["Target"]:
                target_locs.append(loc[0])

    if not text or len(target_locs) == 0:
        return None
    return TriggerIdentificationSample(text=text, trigger_locs=target_locs)


def parse_samples_from_fulltext_doc(doc: Mapping[str, Any]) -> list[TaskSample]:
    """
    Helper to parse sample sentences out of framenet exemplars, contained in lexical units
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu)
    """
    samples: list[TaskSample] = []
    for sentence in doc["sentence"]:
        samples += parse_frame_samples_from_annotation_set(sentence["annotationSet"])
        trigger_sample = parse_trigger_sample_from_annotation_set(
            sentence["annotationSet"]
        )
        if trigger_sample:
            samples.append(trigger_sample)
    return samples

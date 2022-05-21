from __future__ import annotations
from typing import Any, Iterable, Mapping, Optional
from frame_semantic_transformer.data.framenet import get_fulltext_docs

from frame_semantic_transformer.data.sesame_data_splits import (
    SESAME_DEV_FILES,
    SESAME_TEST_FILES,
)
from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    ArgumentsExtractionTask,
    FrameClassificationSample,
    FrameClassificationTask,
    TaskSample,
    TriggerIdentificationSample,
    TriggerIdentificationTask,
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
            for trigger_loc in annotation["Target"]:
                sample_sentences.append(
                    FrameClassificationSample(
                        task=FrameClassificationTask(
                            text=annotation["text"],
                            trigger_loc=trigger_loc[0],
                        ),
                        frame=annotation["frame"]["name"],
                    )
                )
                sample_sentences.append(
                    ArgumentsExtractionSample(
                        task=ArgumentsExtractionTask(
                            text=annotation["text"],
                            trigger_loc=trigger_loc[0],
                            frame=annotation["frame"]["name"],
                        ),
                        frame_element_locs=annotation["FE"][0],
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
    return TriggerIdentificationSample(
        task=TriggerIdentificationTask(text=text), trigger_locs=target_locs
    )


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

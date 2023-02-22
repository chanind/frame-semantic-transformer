from __future__ import annotations
from typing import Any, Iterable, Optional

from nltk.corpus import framenet as fn

from frame_semantic_transformer.data.augmentations import (
    LowercaseAugmentation,
    RemoveContractionsAugmentation,
    RemoveEndPunctuationAugmentation,
)
from frame_semantic_transformer.data.augmentations.DataAugmentation import (
    DataAugmentation,
)
from .ensure_framenet_downloaded import ensure_framenet_downloaded
from .sesame_data_splits import SESAME_DEV_FILES, SESAME_TEST_FILES


from frame_semantic_transformer.data.frame_types import (
    FrameAnnotatedSentence,
    FrameAnnotation,
    FrameElementAnnotation,
)
from ..loader import TrainingLoader


def load_framenet_samples(
    include_docs: Optional[Iterable[str]] = None,
    exclude_docs: Optional[Iterable[str]] = None,
) -> list[FrameAnnotatedSentence]:
    samples: list[FrameAnnotatedSentence] = []
    for doc in fn.docs():
        if exclude_docs and doc["filename"] in exclude_docs:
            continue
        if include_docs and doc["filename"] not in include_docs:
            continue
        samples += parse_annotated_sentences_from_framenet_doc(doc)
    return samples


def parse_annotated_sentences_from_framenet_doc(
    fn_doc: dict[str, Any]
) -> list[FrameAnnotatedSentence]:
    annotated_sentences = []
    for sentence in fn_doc["sentence"]:
        sentence_text = sentence["text"]
        frame_annotations: list[FrameAnnotation] = []
        for fn_annotation in sentence["annotationSet"]:
            if (
                "FE" in fn_annotation
                and "Target" in fn_annotation
                and "frame" in fn_annotation
            ):
                frame_annotations.append(
                    FrameAnnotation(
                        frame=fn_annotation["frame"]["name"],
                        trigger_locs=[loc[0] for loc in fn_annotation["Target"]],
                        frame_elements=[
                            FrameElementAnnotation(
                                start_loc=fn_element[0],
                                end_loc=fn_element[1],
                                name=fn_element[2],
                            )
                            for fn_element in fn_annotation["FE"][0]
                        ],
                    )
                )
        if len(frame_annotations) > 0:
            annotated_sentences.append(
                FrameAnnotatedSentence(
                    text=sentence_text, annotations=frame_annotations
                )
            )
    return annotated_sentences


class Framenet17TrainingLoader(TrainingLoader):
    def setup(self) -> None:
        ensure_framenet_downloaded()

    def get_augmentations(self) -> list[DataAugmentation]:
        return [
            RemoveEndPunctuationAugmentation(0.3),
            LowercaseAugmentation(0.2),
            RemoveContractionsAugmentation(0.2),
        ]

    def load_training_data(self) -> list[FrameAnnotatedSentence]:
        return load_framenet_samples(exclude_docs=SESAME_DEV_FILES + SESAME_TEST_FILES)

    def load_test_data(self) -> list[FrameAnnotatedSentence]:
        return load_framenet_samples(include_docs=SESAME_TEST_FILES)

    def load_validation_data(self) -> list[FrameAnnotatedSentence]:
        return load_framenet_samples(include_docs=SESAME_DEV_FILES)

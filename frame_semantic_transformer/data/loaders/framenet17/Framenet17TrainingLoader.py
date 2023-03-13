from __future__ import annotations
from typing import Any, Iterable, Optional

from nltk.corpus import framenet as fn

from frame_semantic_transformer.data.augmentations import (
    DoubleQuotesAugmentation,
    KeyboardAugmentation,
    LowercaseAugmentation,
    SimpleMisspellingAugmentation,
    RemoveEndPunctuationAugmentation,
    StripPunctuationAugmentation,
    SynonymAugmentation,
    UppercaseAugmentation,
)
from frame_semantic_transformer.data.augmentations.DataAugmentation import (
    DataAugmentation,
)
from frame_semantic_transformer.data.tasks import TriggerIdentificationSample
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


def load_framenet_samples_from_exemplars() -> list[FrameAnnotatedSentence]:
    samples: list[FrameAnnotatedSentence] = []
    # make sure we don't include exemplars if we've already included them in the training data
    all_doc_samples_text = {sample.text for sample in load_framenet_samples()}
    for sent in fn.exemplars():
        annotated_sent = parse_annotated_sentence_from_framenet_sentence(
            sent, skip_trigger_identification_task=True
        )
        if annotated_sent and annotated_sent.text not in all_doc_samples_text:
            samples.append(annotated_sent)
    return samples


def parse_annotated_sentences_from_framenet_doc(
    fn_doc: dict[str, Any]
) -> list[FrameAnnotatedSentence]:
    annotated_sentences = []
    for sentence in fn_doc["sentence"]:
        annotated_sentence = parse_annotated_sentence_from_framenet_sentence(sentence)
        if annotated_sentence:
            annotated_sentences.append(annotated_sentence)
    return annotated_sentences


def parse_annotated_sentence_from_framenet_sentence(
    fn_sentence: dict[str, Any],
    skip_trigger_identification_task: bool = False,
) -> FrameAnnotatedSentence | None:
    sentence_text = fn_sentence["text"]
    frame_annotations: list[FrameAnnotation] = []
    for fn_annotation in fn_sentence["annotationSet"]:
        if (
            "FE" in fn_annotation
            and "Target" in fn_annotation
            and "frame" in fn_annotation
        ):
            trigger_locs = [loc[0] for loc in fn_annotation["Target"]]
            # filter out broken annotations
            for trigger_loc in trigger_locs:
                if trigger_loc >= len(sentence_text):
                    return None
            frame_annotations.append(
                FrameAnnotation(
                    frame=fn_annotation["frame"]["name"],
                    trigger_locs=trigger_locs,
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
        return FrameAnnotatedSentence(
            text=sentence_text,
            annotations=frame_annotations,
            skip_trigger_identification_task=skip_trigger_identification_task,
        )
    return None


class Framenet17TrainingLoader(TrainingLoader):
    include_exemplars: bool

    def __init__(self, include_exemplars: bool = False) -> None:
        super().__init__()
        self.include_exemplars = include_exemplars

    def setup(self) -> None:
        ensure_framenet_downloaded()

    def get_augmentations(self) -> list[DataAugmentation]:
        return [
            RemoveEndPunctuationAugmentation(0.5),
            DoubleQuotesAugmentation(0.2),
            StripPunctuationAugmentation(0.2),
            SynonymAugmentation(
                lambda sample: 0.2
                if isinstance(sample, TriggerIdentificationSample)
                else 0.05
            ),
            KeyboardAugmentation(
                lambda sample: 0.3
                if isinstance(sample, TriggerIdentificationSample)
                else 0.05
            ),
            SimpleMisspellingAugmentation(
                lambda sample: 0.3
                if isinstance(sample, TriggerIdentificationSample)
                else 0.05
            ),
            LowercaseAugmentation(0.1),
            UppercaseAugmentation(0.1),
        ]

    def load_training_data(self) -> list[FrameAnnotatedSentence]:
        training_samples = load_framenet_samples(
            exclude_docs=SESAME_DEV_FILES + SESAME_TEST_FILES
        )
        if self.include_exemplars:
            training_samples += load_framenet_samples_from_exemplars()
        return training_samples

    def load_test_data(self) -> list[FrameAnnotatedSentence]:
        return load_framenet_samples(include_docs=SESAME_TEST_FILES)

    def load_validation_data(self) -> list[FrameAnnotatedSentence]:
        return load_framenet_samples(include_docs=SESAME_DEV_FILES)

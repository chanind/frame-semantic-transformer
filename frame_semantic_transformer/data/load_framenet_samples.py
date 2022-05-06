from __future__ import annotations
from nltk.corpus import framenet as fn
from typing import Any, Iterable, Mapping, Optional

from frame_semantic_transformer.data.sesame_data_splits import SESAME_TEST_FILES

from .SampleSentence import SampleSentence


def load_framenet_samples(
    exclude_docs: Optional[Iterable[str]] = None,
) -> list[SampleSentence]:
    samples: list[SampleSentence] = []
    for lu in fn.lus():
        samples += parse_samples_from_lexical_unit(lu)
    for doc in fn.docs():
        if exclude_docs and doc.filename in exclude_docs:
            continue
        samples += parse_samples_from_fulltext_doc(doc)
    return samples


def load_sesame_test_samples() -> list[SampleSentence]:
    samples: list[SampleSentence] = []
    for doc in fn.docs():
        if doc.filename in SESAME_TEST_FILES:
            continue
        samples += parse_samples_from_fulltext_doc(doc)
    return samples


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


def parse_samples_from_annotation_set(
    annotation_set: Iterable[Mapping[str, Any]]
) -> list[SampleSentence]:
    """
    Helper to parse sample sentences out of framenet annotation sets
    Not all annotation sets contain frames, so this will filter out any invalid annotation sets
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu.exemplars)
    """
    sample_sentences: list[SampleSentence] = []
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
                SampleSentence(
                    text=annotation["text"],
                    trigger_loc=trigger_loc,
                    frame=annotation["frame"]["name"],
                    frame_element_locs=annotation["FE"][0],
                )
            )
    return sample_sentences


def parse_samples_from_lexical_unit(
    lexical_unit: Mapping[str, Any]
) -> list[SampleSentence]:
    """
    Helper to parse sample sentences out of framenet exemplars, contained in lexical units
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu)
    """
    sample_sentences: list[SampleSentence] = []
    for exemplar in lexical_unit["exemplars"]:
        sample_sentences += parse_samples_from_annotation_set(exemplar["annotationSet"])
    return sample_sentences


def parse_samples_from_fulltext_doc(doc: Mapping[str, Any]) -> list[SampleSentence]:
    """
    Helper to parse sample sentences out of framenet exemplars, contained in lexical units
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu)
    """
    sample_sentences: list[SampleSentence] = []
    for sentence in doc["sentence"]:
        sample_sentences += parse_samples_from_annotation_set(sentence["annotationSet"])
    return sample_sentences

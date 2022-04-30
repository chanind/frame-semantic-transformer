from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass
class SampleSentence:
    text: str
    trigger_loc: tuple[int, int]
    frame: str
    frame_element_locs: list[tuple[int, int, str]]

    @property
    def trigger(self) -> str:
        return self.text[self.trigger_loc[0] : self.trigger_loc[1]]

    @property
    def trigger_labeled_text(self) -> str:
        pre_span = self.text[0 : self.trigger_loc[0]]
        post_span = self.text[self.trigger_loc[1] :]
        return f"{pre_span}* {self.trigger} *{post_span}"

    @property
    def frame_elements(self) -> list[tuple[str, str]]:
        return [
            (element, self.text[loc_start:loc_end])
            for (loc_start, loc_end, element) in self.frame_element_locs
        ]

    @property
    def frame_elements_str(self) -> str:
        return " | ".join(
            [f"{element} = {text}" for element, text in self.frame_elements]
        )


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

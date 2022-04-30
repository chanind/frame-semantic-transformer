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
        return f"{pre_span} * {self.trigger} * {post_span}"

    @property
    def frame_elements(self) -> list[tuple[str, str]]:
        return [
            (element, self.text[loc_start:loc_end])
            for (loc_start, loc_end, element) in self.frame_element_locs
        ]


def parse_samples_from_exemplars(
    exemplars: Iterable[Mapping[str, Any]]
) -> list[SampleSentence]:
    """
    Helper to parse sample sentences out of framenet exemplars, contained in lexical units
    ex: lu = fn.lus()[0]; samples = parse_samples_from_exemplars(lu.exemplars)
    """
    sample_sentences: list[SampleSentence] = []
    for exemplar in exemplars:
        for annotation in exemplar["annotationSet"]:
            if "FE" in annotation and "Target" in annotation and "frame" in annotation:
                assert len(annotation["Target"]) == 1
                assert annotation["FE"][1] == {}
                sample_sentences.append(
                    SampleSentence(
                        text=annotation["text"],
                        trigger_loc=annotation["Target"][0],
                        frame=annotation["frame"]["name"],
                        frame_element_locs=annotation["FE"][0],
                    )
                )
    return sample_sentences

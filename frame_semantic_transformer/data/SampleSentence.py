from __future__ import annotations
from dataclasses import dataclass


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
    def frame_classification_input(self) -> str:
        return f"FRAME: {self.trigger_labeled_text}"

    @property
    def frame_args_input(self) -> str:
        return f"ARGS {self.frame}: {self.trigger_labeled_text}"

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

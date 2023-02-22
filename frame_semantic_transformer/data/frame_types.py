from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Frame:
    """
    Representation of a FrameNet frame
    For training on your own data, you can use this class to represent your own frames
    """

    name: str
    core_elements: list[str]
    non_core_elements: list[str]
    lexical_units: list[str]


@dataclass
class FrameAnnotatedSentence:
    """
    Representation of a sentence with annotations for use in training
    If training on your own data, you'll need to create instances of this class for your training sentences
    """

    text: str
    annotations: list[FrameAnnotation]


@dataclass
class FrameAnnotation:
    """
    A single frame occuring in a sentence
    """

    frame: str
    trigger_locs: list[int]
    frame_elements: list[FrameElementAnnotation]


@dataclass
class FrameElementAnnotation:
    """
    A single frame element in a frame annotation.
    Includes the name of the frame element and the start and end locations of the frame element in the sentence
    """

    name: str
    start_loc: int
    end_loc: int

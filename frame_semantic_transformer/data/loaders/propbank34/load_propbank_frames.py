from __future__ import annotations

import nltk
from glob import glob
from os import path
from xml.etree import ElementTree


from frame_semantic_transformer.data.frame_types import Frame


def load_propbank_frames() -> list[Frame]:
    """
    Load the full list of frames to be used during inference
    """
    dataset_path = nltk.data.find("corpora/propbank-frames-3.4.0").path
    frames_paths = glob(path.join(dataset_path, "frames", "*.xml"))
    frames = []
    for frame_path in frames_paths:
        with open(frame_path, "r") as frame_file:
            etree = ElementTree.parse(frame_file).getroot()
            raw_frames = etree.findall("predicate/roleset")
            for raw_frame in raw_frames:
                frame = Frame(
                    name=raw_frame.attrib["id"],
                    core_elements=[
                        f"ARG{role.attrib['n']}-{role.attrib['f']}"
                        for role in raw_frame.findall("roles/role")
                    ],
                    non_core_elements=[],
                    lexical_units=[
                        f"{alias.text}.{alias.attrib['pos']}"
                        for alias in raw_frame.findall("aliases/alias")
                    ],
                )
                frames.append(frame)
    return frames

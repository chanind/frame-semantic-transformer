from __future__ import annotations
import re
import nltk
from glob import glob
from os import path
from xml.etree import ElementTree

from nltk.stem import PorterStemmer

from .ensure_propbank_downloaded import ensure_propbank_downloaded


from frame_semantic_transformer.data.frame_types import Frame
from ..loader import InferenceLoader


base_stemmer = PorterStemmer()


class Propbank31InferenceLoader(InferenceLoader):
    """
    Inference loader for Propbank 3.1 data
    """

    def setup(self) -> None:
        ensure_propbank_downloaded()

    def strict_frame_elements(self) -> bool:
        """
        Propbank only lists core roles, not all roles, so we can't enforce strict frame elements
        """
        return False

    def load_frames(self) -> list[Frame]:
        """
        Load the full list of frames to be used during inference
        """
        dataset_path = nltk.data.find("corpora/propbank-frames-3.1").path
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

    def normalize_lexical_unit_text(self, lu: str) -> str:
        """
        Normalize a lexical unit like "takes.v" to "take".
        """
        normalized_lu = lu.lower()
        normalized_lu = re.sub(r"\.[a-zA-Z]+$", "", normalized_lu)
        normalized_lu = re.sub(r"[^a-z0-9 ]", "", normalized_lu)
        return base_stemmer.stem(normalized_lu.strip())

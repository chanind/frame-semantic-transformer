from __future__ import annotations
import re

from nltk.stem import PorterStemmer

from .load_propbank_frames import load_propbank_frames

from .ensure_propbank_downloaded import ensure_propbank_downloaded


from frame_semantic_transformer.data.frame_types import Frame
from ..loader import InferenceLoader


base_stemmer = PorterStemmer()

LOW_PRIORITY_LONGER_LUS = {"back", "down", "make", "take", "have", "into", "come"}


class Propbank34InferenceLoader(InferenceLoader):
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
        return load_propbank_frames()

    def normalize_lexical_unit_text(self, lu: str) -> str:
        """
        Normalize a lexical unit like "takes.v" to "take".
        """
        normalized_lu = lu.lower().replace("_", " ")
        normalized_lu = re.sub(r"\.[a-zA-Z]+$", "", normalized_lu)
        normalized_lu = re.sub(r"[^a-z0-9 ]", "", normalized_lu)
        return base_stemmer.stem(normalized_lu.strip())

    def prioritize_lexical_unit(self, lu: str) -> bool:
        """
        Check if the lexical unit is relatively rare, so that it should be considered "high information"
        """
        norm_lu = self.normalize_lexical_unit_text(lu)
        return len(norm_lu) >= 4 and norm_lu not in LOW_PRIORITY_LONGER_LUS

from __future__ import annotations
import re

from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from nltk.corpus import framenet as fn

from .ensure_framenet_downloaded import ensure_framenet_downloaded
from .ensure_wordnet_downloaded import ensure_wordnet_downloaded
from frame_semantic_transformer.data.frame_types import Frame
from ..loader import InferenceLoader


porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer("english")
wordnet_lemmatizer = WordNetLemmatizer()

WORDNET_LEMMATIZER_POS = ["a", "r", "n", "v", "s"]


LOW_PRIORITY_LONGER_LUS = {"back", "down", "make", "take", "have", "into", "come"}


class Framenet17InferenceLoader(InferenceLoader):
    """
    Inference loader for FrameNet 1.7 data
    """

    def setup(self) -> None:
        ensure_framenet_downloaded()
        ensure_wordnet_downloaded()

    def load_frames(self) -> list[Frame]:
        """
        Load the full list of frames to be used during inference
        """
        frames = []
        for raw_frame in fn.frames():
            frame = Frame(
                name=raw_frame.name,
                core_elements=[
                    name for (name, fe) in raw_frame.FE.items() if fe.coreType == "Core"
                ],
                non_core_elements=[
                    name for (name, fe) in raw_frame.FE.items() if fe.coreType != "Core"
                ],
                lexical_units=[lu for lu in raw_frame.lexUnit.keys()],
            )
            frames.append(frame)
        return frames

    def normalize_lexical_unit_text(self, lu: str) -> str | set[str]:
        """
        Normalize a lexical unit like "takes.v" to "take".
        """
        normalized_lu = lu.lower()
        normalized_lu = re.sub(r"\.[a-zA-Z]+$", "", normalized_lu)
        normalized_lu = re.sub(r"[^a-z0-9 ]", "", normalized_lu)
        normalized_lu = normalized_lu.strip()
        norm_lus = {
            porter_stemmer.stem(normalized_lu),
            lancaster_stemmer.stem(normalized_lu),
            snowball_stemmer.stem(normalized_lu),
        }
        # try every possible part of speech for the wordnet lemmatizer
        for pos in WORDNET_LEMMATIZER_POS:
            norm_lus.add(wordnet_lemmatizer.lemmatize(normalized_lu, pos=pos))
        return norm_lus

    def prioritize_lexical_unit(self, lu: str) -> bool:
        """
        Check if the lexical unit is relatively rare, so that it should be considered "high information"
        """
        norm_lu = self.normalize_lexical_unit_text(lu)
        return len(norm_lu) >= 4 and norm_lu not in LOW_PRIORITY_LONGER_LUS

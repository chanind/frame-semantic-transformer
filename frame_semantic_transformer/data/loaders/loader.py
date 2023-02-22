from __future__ import annotations
from abc import ABC, abstractmethod

from frame_semantic_transformer.data.augmentations.DataAugmentation import (
    DataAugmentation,
)
from frame_semantic_transformer.data.frame_types import Frame, FrameAnnotatedSentence


class InferenceLoader(ABC):
    """
    Base class for all inference loaders
    """

    def name(self) -> str:
        """
        Return a name for this loader.
        Frame-Semantic-Transformer will enforce that a loader with this name is used at inference time.
        """
        return self.__class__.__name__

    def setup(self) -> None:
        """
        Perform any setup required, e.g. downloading needed data
        """
        pass

    @abstractmethod
    def load_frames(self) -> list[Frame]:
        """
        Load the full list of frames to be used during inference
        """
        pass

    @abstractmethod
    def normalize_lexical_unit_text(self, lu: str) -> str:
        """
        Normalize a lexical unit like "takes.v" to "take".
        """
        pass

    def prioritize_lexical_unit(self, lu: str) -> bool:
        """
        Check if the lexical unit is relatively rare, so that it should be considered "high information"
        """
        return len(self.normalize_lexical_unit_text(lu)) >= 6


class TrainingLoader(ABC):
    """
    Base class for all training loaders
    """

    def name(self) -> str:
        """
        Return a name for this loader.
        Frame-Semantic-Transformer will enforce that a loader with this name is used at eval time.
        """
        return self.__class__.__name__

    def setup(self) -> None:
        """
        Perform any setup required, e.g. downloading needed data.
        """
        pass

    @abstractmethod
    def get_augmentations(self) -> list[DataAugmentation]:
        """
        Get a list of augmentations to apply to the training data
        """
        pass

    @abstractmethod
    def load_training_data(self) -> list[FrameAnnotatedSentence]:
        """
        Load the training data
        """
        pass

    @abstractmethod
    def load_validation_data(self) -> list[FrameAnnotatedSentence]:
        """
        Load the validation data
        """
        pass

    @abstractmethod
    def load_test_data(self) -> list[FrameAnnotatedSentence]:
        """
        Load the test data
        """
        pass

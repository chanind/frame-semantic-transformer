from __future__ import annotations
from syrupy.assertion import SnapshotAssertion

from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17TrainingLoader,
    Framenet17InferenceLoader,
)
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)

training_loader = Framenet17TrainingLoader()
loader_cache = LoaderDataCache(Framenet17InferenceLoader())


def test_tasks_from_annotated_sentences(snapshot: SnapshotAssertion) -> None:
    sentences = training_loader.load_test_data()
    # random sentences
    samples = tasks_from_annotated_sentences(sentences[11:37], loader_cache)
    assert samples == snapshot

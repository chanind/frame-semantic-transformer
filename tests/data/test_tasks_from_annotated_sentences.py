from __future__ import annotations
from syrupy.assertion import SnapshotAssertion

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.loaders.loader import TrainingLoader
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)


def test_tasks_from_annotated_sentences(
    loader_cache: LoaderDataCache,
    training_loader: TrainingLoader,
    snapshot: SnapshotAssertion,
) -> None:
    sentences = training_loader.load_test_data()
    # random sentences
    samples = tasks_from_annotated_sentences(sentences[11:37], loader_cache)
    assert [
        (s.get_task_name(), s.get_input(), s.get_target()) for s in samples
    ] == snapshot

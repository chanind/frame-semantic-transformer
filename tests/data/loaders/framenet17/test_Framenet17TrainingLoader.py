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


def test_load_sesame_test_samples() -> None:
    sentences = training_loader.load_test_data()
    samples = tasks_from_annotated_sentences(sentences, loader_cache)
    assert len(samples) == 15126


def test_load_sesame_dev_samples() -> None:
    sentences = training_loader.load_validation_data()
    samples = tasks_from_annotated_sentences(sentences, loader_cache)
    assert len(samples) == 5166


def test_load_sesame_train_samples() -> None:
    sentences = training_loader.load_training_data()
    samples = tasks_from_annotated_sentences(sentences, loader_cache)
    trigger_id_samples = [
        sample
        for sample in samples
        if sample.get_task_name() == "trigger_identification"
    ]
    frame_id_samples = [
        sample for sample in samples if sample.get_task_name() == "frame_classification"
    ]
    assert len(trigger_id_samples) == 3425
    assert len(frame_id_samples) == 20597
    assert len(samples) == 44619

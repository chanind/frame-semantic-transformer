from __future__ import annotations
from transformers import T5Tokenizer
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.TaskSampleDataset import (
    TaskSampleDataset,
    balance_tasks_by_type,
)
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    ArgumentsExtractionTask,
    FrameClassificationSample,
    FrameClassificationTask,
    TriggerIdentificationSample,
    TriggerIdentificationTask,
)

training_loader = Framenet17TrainingLoader()
loader_cache = LoaderDataCache(Framenet17InferenceLoader())


def test_TaskSampleDataset() -> None:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    sentences = training_loader.load_training_data()
    # use the first 8 samples
    samples = tasks_from_annotated_sentences(sentences, loader_cache)[0:8]

    dataset = TaskSampleDataset(samples, tokenizer)

    assert len(dataset) == 8
    assert len(dataset[0]["input_ids"]) == 512
    assert len(dataset[0]["attention_mask"]) == 512
    assert len(dataset[0]["labels"]) == 512


def test_balance_tasks_by_type() -> None:
    tasks = [
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a1", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a2", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a3", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a4", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a5", 0, "Greetings", loader_cache), []
        ),
        FrameClassificationSample(
            FrameClassificationTask("f1", 0, loader_cache), "Greetings"
        ),
        FrameClassificationSample(
            FrameClassificationTask("f2", 0, loader_cache), "Greetings"
        ),
        TriggerIdentificationSample(TriggerIdentificationTask("t1"), []),
    ]
    balanced_tasks = balance_tasks_by_type(tasks, max_duplication_factor=10)
    assert len(balanced_tasks) == 14  # 5 arg, 4 frame, 5 trigger
    assert (
        len([t for t in balanced_tasks if t.get_task_name() == "args_extraction"]) == 5
    )
    assert (
        len([t for t in balanced_tasks if t.get_task_name() == "frame_classification"])
        == 4
    )
    assert (
        len(
            [t for t in balanced_tasks if t.get_task_name() == "trigger_identification"]
        )
        == 5
    )


def test_balance_tasks_by_type_caps_replications() -> None:
    tasks = [
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a1", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a2", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a3", 0, "Greetings", loader_cache), []
        ),
        ArgumentsExtractionSample(
            ArgumentsExtractionTask("a4", 0, "Greetings", loader_cache), []
        ),
        TriggerIdentificationSample(TriggerIdentificationTask("t1"), []),
    ]
    capped_balanced_tasks = balance_tasks_by_type(tasks, max_duplication_factor=2)
    assert len(capped_balanced_tasks) == 6  # 4 arg, 2 trigger

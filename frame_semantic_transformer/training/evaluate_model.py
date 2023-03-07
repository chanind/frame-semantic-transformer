from __future__ import annotations
from typing import Optional

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from frame_semantic_transformer.constants import DEFAULT_NUM_WORKERS

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.loaders.loader import (
    InferenceLoader,
    TrainingLoader,
)
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from .TrainingModelWrapper import TrainingModelWrapper


def evaluate_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    batch_size: int = 8,
    num_workers: int = DEFAULT_NUM_WORKERS,
    inference_loader: Optional[InferenceLoader] = None,
    training_loader: Optional[TrainingLoader] = None,
    log_eval_failures: bool = False,
) -> None:
    """
    Benchmark this model against the validation and test sets
    """

    if not inference_loader:
        inference_loader = Framenet17InferenceLoader()
    loader_cache = LoaderDataCache(inference_loader)
    if not training_loader:
        training_loader = Framenet17TrainingLoader()

    inference_loader.setup()
    training_loader.setup()

    config = model.config
    expected_inference_loader = (
        config.inference_loader
        if hasattr(config, "inference_loader")
        else inference_loader.name()
    )
    expected_training_loader = (
        config.training_loader
        if hasattr(config, "training_loader")
        else training_loader.name()
    )

    if expected_inference_loader != inference_loader.name():
        raise ValueError(
            f"Model was trained with inference loader {expected_inference_loader} but is being evaluated with {inference_loader.name()}"
        )
    if expected_training_loader != training_loader.name():
        raise ValueError(
            f"Model was trained with training loader {expected_training_loader} but is being evaluated with {training_loader.name()}"
        )

    model_wrapper = TrainingModelWrapper(
        model, tokenizer, loader_cache, log_eval_failures=log_eval_failures
    )
    trainer = Trainer(gpus=1, precision=32, max_epochs=1)

    val_dataset = TaskSampleDataset(
        tasks_from_annotated_sentences(
            training_loader.load_validation_data(), loader_cache
        ),
        tokenizer,
        balance_tasks=False,
    )
    test_dataset = TaskSampleDataset(
        tasks_from_annotated_sentences(training_loader.load_test_data(), loader_cache),
        tokenizer,
        balance_tasks=False,
    )

    with torch.no_grad():
        print("Evaluating on validation set")
        trainer.validate(
            model_wrapper,
            dataloaders=DataLoader(
                val_dataset, batch_size=batch_size, num_workers=num_workers
            ),
        )

        print("Evaluating on test set")
        trainer.test(
            model_wrapper,
            dataloaders=DataLoader(
                test_dataset, batch_size=batch_size, num_workers=num_workers
            ),
        )

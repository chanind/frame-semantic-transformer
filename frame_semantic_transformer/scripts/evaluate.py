from __future__ import annotations
import argparse
import logging
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from frame_semantic_transformer.training import TrainingModelWrapper

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic transformer model using FrameNet 1.7"
    )
    parser.add_argument(
        "--model-path",
        help="The path to the model to evaluate, or the name of a pretrained huggingface model",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="default 2")
    parser.add_argument("--batch-size", type=int, default=8, help="default 8")
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    training_loader = Framenet17TrainingLoader()
    inference_loader = Framenet17InferenceLoader()
    loader_cache = LoaderDataCache(inference_loader)

    inference_loader.setup()
    training_loader.setup()

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

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

    model_wrapper = TrainingModelWrapper(model, tokenizer, loader_cache)
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
                val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            ),
        )

        print("Evaluating on test set")
        trainer.test(
            model_wrapper,
            dataloaders=DataLoader(
                test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            ),
        )

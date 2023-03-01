from __future__ import annotations
import argparse
import logging
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.training import evaluate_model

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

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

    evaluate_model(
        model,
        tokenizer,
        loader_cache=loader_cache,
        training_loader=training_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

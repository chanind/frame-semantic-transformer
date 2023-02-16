from __future__ import annotations
import logging
import os
import argparse

from frame_semantic_transformer.training.train import train

DEFAULT_NUM_WORKERS = os.cpu_count() or 2
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a new frame semantic transformer model using FrameNet 1.7"
    )
    parser.add_argument(
        "--base-model",
        default="t5-base",
        help="The HuggingFace T5 model to use as a starting point, default t5-base",
    )
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--batch-size", type=int, default=8, help="default 8")
    parser.add_argument("--epochs", type=int, default=10, help="default 10")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="dir where output models will be saved after each epoch, default ./outputs",
    )
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    train(
        base_model=args.base_model,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
    )

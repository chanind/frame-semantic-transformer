from __future__ import annotations
import argparse
from collections import defaultdict
import logging
from typing import Any, Sequence, Type
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch

from frame_semantic_transformer.constants import PADDING_LABEL_ID
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import chunk_list
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.tasks.ArgumentsExtractionSample import (
    ArgumentsExtractionSample,
)
from frame_semantic_transformer.data.tasks.FrameClassificationSample import (
    FrameClassificationSample,
)
from frame_semantic_transformer.data.tasks.TaskSample import TaskSample
from frame_semantic_transformer.data.tasks.TriggerIdentificationSample import (
    TriggerIdentificationSample,
)
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from frame_semantic_transformer.predict import batch_predict, predict_on_ids

logger = logging.getLogger(__name__)


def calc_eval_metrics(
    true_pos: float, false_pos: float, false_neg: float
) -> dict[str, float]:
    """
    Calculate precision, recall, and f score
    Based on https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py
    """
    if true_pos == 0 and false_pos == 0:
        precision = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos == 0 and false_neg == 0:
        recall = 0.0
    else:
        recall = true_pos / (true_pos + false_neg)
    if precision == 0 and recall == 0:
        f_score = 0.0
    else:
        f_score = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f_score}


def evaluate(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    samples: Sequence[TaskSample],
    loader_cache: LoaderDataCache,
    batch_size: int = 10,
    print_failures: bool = False,
    predictions_per_sample: int = 5,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 2.5,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0])
    for samples_chunk in tqdm(
        chunk_list(samples, batch_size), total=len(samples) / batch_size
    ):
        inputs = [sample.get_input() for sample in samples_chunk]

        predictions = batch_predict(
            model,
            tokenizer,
            inputs,
            num_beams=predictions_per_sample,
            num_return_sequences=predictions_per_sample,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        batched_predictions = chunk_list(predictions, predictions_per_sample)
        for sample, preds in zip(samples_chunk, batched_predictions):
            assert len(preds) == predictions_per_sample
            score = sample.evaluate_prediction(
                preds, sample.get_target(), sample.get_input(), loader_cache
            )
            true_pos, false_pos, false_neg = score
            results[sample.get_task_name()][0] += true_pos
            results[sample.get_task_name()][1] += false_pos
            results[sample.get_task_name()][2] += false_neg
            if print_failures and (false_neg > 0 or false_pos > 0):
                print(score)
                print(sample.get_target())
                print(preds)
                print("\n")

    return results


# TODO: figure out a better way to lookup the class from the string
TASK_SAMPLE_CLASS_MAP: dict[str, Type[TaskSample]] = {
    "args_extraction": ArgumentsExtractionSample,
    "frame_classification": FrameClassificationSample,
    "trigger_identification": TriggerIdentificationSample,
}


def evaluate_batch(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    batch: Any,
    loader_cache: LoaderDataCache,
    predictions_per_sample: int = 5,
) -> dict[str, list[float]]:
    predictions = predict_on_ids(
        model,
        tokenizer,
        batch["input_ids"],
        batch["attention_mask"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        num_beams=predictions_per_sample,
        num_return_sequences=predictions_per_sample,
    )
    batched_predictions = chunk_list(predictions, predictions_per_sample)
    results: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0])
    for preds, task, label, input_ids in zip(
        batched_predictions, batch["task"], batch["labels"], batch["input_ids"]
    ):
        assert len(preds) == predictions_per_sample
        target_tokens = [
            tok_id for tok_id in label.tolist() if tok_id != PADDING_LABEL_ID
        ]
        input_tokens = [
            tok_id for tok_id in input_ids.tolist() if tok_id != PADDING_LABEL_ID
        ]
        target = tokenizer.decode(
            target_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        input = tokenizer.decode(
            input_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        sample_class = TASK_SAMPLE_CLASS_MAP[task]
        true_pos, false_pos, false_neg = sample_class.evaluate_prediction(
            preds, target, input, loader_cache
        )
        results[task][0] += true_pos
        results[task][1] += false_pos
        results[task][2] += false_neg

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic transformer model using FrameNet 1.7"
    )
    parser.add_argument(
        "--base-model",
        default="t5-base",
        help="The HuggingFace T5 model to use as a starting point, default t5-base",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="default 8")
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    training_loader = Framenet17TrainingLoader()
    inference_loader = Framenet17InferenceLoader()
    loader_cache = LoaderDataCache(inference_loader)

    inference_loader.setup()
    training_loader.setup()

    device = torch.device("cuda" if args.use_gpu else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.base_model).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(args.base_model)

    with torch.no_grad():
        print("Evaluating on validation set")
        validation_samples = tasks_from_annotated_sentences(
            training_loader.load_validation_data(), loader_cache
        )
        validation_results = evaluate(
            model,
            tokenizer,
            validation_samples,
            loader_cache,
            batch_size=args.batch_size,
        )
        for task, metrics in validation_results.items():
            scores = calc_eval_metrics(*metrics)
            print(f"{task}: {scores}")

        print("Evaluating on test set")
        test_samples = tasks_from_annotated_sentences(
            training_loader.load_test_data(), loader_cache
        )
        test_results = evaluate(
            model,
            tokenizer,
            test_samples,
            loader_cache,
            batch_size=args.batch_size,
        )
        for task, metrics in test_results.items():
            scores = calc_eval_metrics(*metrics)
            print(f"{task}: {scores}")

from __future__ import annotations
from collections import defaultdict
from typing import Sequence
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from frame_semantic_transformer.data.data_utils import chunk_list
from frame_semantic_transformer.data.task_samples.TaskSample import TaskSample
from frame_semantic_transformer.predict import batch_predict


def calc_eval_metrics(
    true_pos: int, false_pos: int, false_neg: int
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
    tokenizer: T5Tokenizer,
    samples: Sequence[TaskSample],
    batch_size: int = 10,
) -> dict[str, list[int]]:
    results: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for samples_chunk in tqdm(
        chunk_list(samples, batch_size), total=len(samples) / batch_size
    ):
        inputs = [sample.get_input() for sample in samples_chunk]

        predictions = batch_predict(model, tokenizer, inputs)
        for sample, prediction in zip(samples_chunk, predictions):
            true_pos, false_pos, false_neg = sample.evaluate_prediction(prediction)
            results[sample.get_task_name()][0] += true_pos
            results[sample.get_task_name()][1] += false_pos
            results[sample.get_task_name()][2] += false_neg
    return results

from __future__ import annotations
from collections import defaultdict
import logging
from typing import Any, Type
from dataclasses import asdict, dataclass, field
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.constants import PADDING_LABEL_ID
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import chunk_list
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
from frame_semantic_transformer.predict import predict_on_ids

logger = logging.getLogger(__name__)


def calc_eval_metrics(scores: EvalScores) -> dict[str, float]:
    """
    Calculate precision, recall, and f score
    Based on https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py
    """
    if scores.true_pos == 0 and scores.false_pos == 0:
        precision = 0.0
    else:
        precision = scores.true_pos / (scores.true_pos + scores.false_pos)
    if scores.true_pos == 0 and scores.false_neg == 0:
        recall = 0.0
    else:
        recall = scores.true_pos / (scores.true_pos + scores.false_neg)
    if precision == 0 and recall == 0:
        f_score = 0.0
    else:
        f_score = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f_score}


# TODO: figure out a better way to lookup the class from the string
TASK_SAMPLE_CLASS_MAP: dict[str, Type[TaskSample]] = {
    "args_extraction": ArgumentsExtractionSample,
    "frame_classification": FrameClassificationSample,
    "trigger_identification": TriggerIdentificationSample,
}


@dataclass
class EvalScores:
    true_pos: float = 0
    false_pos: float = 0
    false_neg: float = 0


@dataclass
class EvalFailure:
    input: str
    target: str
    predictions: list[str]


@dataclass
class TaskEvalResults:
    scores: EvalScores = field(default_factory=EvalScores)
    false_positives: list[EvalFailure] = field(default_factory=list)
    false_negatives: list[EvalFailure] = field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        return {
            "scores": asdict(self.scores),
            "false_positives": list(map(asdict, self.false_positives)),
            "false_negatives": list(map(asdict, self.false_negatives)),
        }


def evaluate_batch(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    batch: Any,
    loader_cache: LoaderDataCache,
    predictions_per_sample: int = 5,
) -> dict[str, TaskEvalResults]:
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
    results: dict[str, TaskEvalResults] = defaultdict(TaskEvalResults)
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
        results[task].scores.true_pos += true_pos
        results[task].scores.false_pos += false_pos
        results[task].scores.false_neg += false_neg
        if false_pos > 0:
            results[task].false_positives.append(
                EvalFailure(input=input, target=target, predictions=list(preds))
            )
        if false_neg > 0:
            results[task].false_negatives.append(
                EvalFailure(input=input, target=target, predictions=list(preds))
            )

    return results

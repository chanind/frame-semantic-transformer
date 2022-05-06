from __future__ import annotations
from typing import Iterable
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.corpus import framenet as fn

from frame_semantic_transformer.data.SampleSentence import SampleSentence
from frame_semantic_transformer.predict import predict


all_valid_frames = {frame.name for frame in fn.frames()}


def evaluate(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    samples: Iterable[SampleSentence],
) -> dict[str, list[int]]:
    results: dict[str, list[int]] = {"frame": [0, 0, 0], "args": [0, 0, 0]}
    for sample in tqdm(samples):
        frame_task_input = sample.frame_classification_input
        args_task_input = sample.frame_args_input

        frame_prediction = predict(model, tokenizer, frame_task_input)[0]
        args_prediction = predict(model, tokenizer, args_task_input)[0]

        if frame_prediction == sample.frame:
            results["frame"][0] += 1
        elif frame_prediction in all_valid_frames:
            results["frame"][1] += 1
        else:
            results["frame"][2] += 1

        if args_prediction == sample.frame_elements_str:
            results["args"][0] += 1
        # TODO: figure out fp/fn for frame elements
        else:
            results["args"][1] += 1
    return results

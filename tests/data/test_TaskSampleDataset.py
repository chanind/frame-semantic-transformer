from __future__ import annotations
from transformers import T5Tokenizer
from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
from frame_semantic_transformer.data.framenet import get_fulltext_docs

from frame_semantic_transformer.data.load_framenet_samples import (
    parse_samples_from_fulltext_doc,
)


def test_TaskSampleDataset() -> None:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    doc = get_fulltext_docs()[1]
    # use the first 8 samples
    samples = parse_samples_from_fulltext_doc(doc)[0:8]

    dataset = TaskSampleDataset(samples, tokenizer)

    assert len(dataset) == 8
    assert len(dataset[0]["input_ids"]) == 55
    assert len(dataset[0]["attention_mask"]) == 55
    assert len(dataset[0]["labels"]) == 30

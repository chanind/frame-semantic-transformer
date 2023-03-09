from __future__ import annotations
from collections import defaultdict

from os import path
from glob import glob
import re

from nltk.corpus.reader.conll import ConllCorpusReader

from frame_semantic_transformer.data.augmentations import (
    KeyboardAugmentation,
    LowercaseAugmentation,
    RemoveEndPunctuationAugmentation,
    SimpleMisspellingAugmentation,
    SynonymAugmentation,
    UppercaseAugmentation,
)
from frame_semantic_transformer.data.augmentations.DataAugmentation import (
    DataAugmentation,
)

from frame_semantic_transformer.data.frame_types import (
    FrameAnnotatedSentence,
    FrameAnnotation,
    FrameElementAnnotation,
)
from ..loader import TrainingLoader
from .load_propbank_frames import load_propbank_frames


SPLITS = {
    "train": [
        "docs/evaluation/ewt.dev.txt",
        "docs/evaluation/ontonotes-train-list.txt",
    ],
    "val": ["docs/evaluation/ewt.dev.txt", "docs/evaluation/ontonotes-dev-list.txt"],
    "test": ["docs/evaluation/ewt.test.txt", "docs/evaluation/ontonotes-test-list.txt"],
}
EWT_GLOB = "data/google/ewt/**/*.gold_conll"
ONTONOTES_GLOB = "data/ontonotes/**/*.gold_conll"


def load_docs_set(base_path: str, docs_list_paths: list[str]) -> list[str]:
    docs_lookup = set()
    for docs_list_path in docs_list_paths:
        with open(path.join(base_path, docs_list_path)) as f:
            raw_docs = f.read().splitlines()
            # weirdly the ewt dev files end in .conllu but nothing else does
            docs_lookup.update([doc.replace(".conllu", "") for doc in raw_docs])

    docs = []
    ewt_docs = glob(path.join(base_path, EWT_GLOB), recursive=True)
    for doc in ewt_docs:
        doc_base = re.sub(r".*/data/google/ewt/", "", doc).replace(".gold_conll", "")
        if doc_base in docs_lookup:
            docs.append(doc)

    ontonotes_docs = glob(path.join(base_path, ONTONOTES_GLOB), recursive=True)
    for doc in ontonotes_docs:
        # for some reason the ontonotes list has 'ontonotes' in the path, but ewt doesn't have 'google/ewt'
        doc_base = re.sub(r".*/data/ontonotes/", "ontonotes/", doc).replace(
            ".gold_conll", ""
        )
        if doc_base in docs_lookup:
            docs.append(doc)
    return docs


def conll_word_index_to_locs(words: list[str], word_index: int) -> tuple[int, int]:
    """
    Take a list of words and an index of a word and return the start and end char indices of the word in the sentence
    """
    start_loc = 0
    for i, word in enumerate(words):
        if i == word_index:
            return start_loc, start_loc + len(word)
        start_loc += len(word) + 1
    raise ValueError("word index out of range")


def load_propbank_samples(
    docs_list: list[str], valid_frames: set[str]
) -> list[FrameAnnotatedSentence]:
    """
    Parse each of the propbank ontonotes and ewt gold conll files and return a list of FrameAnnotatedSentence objects
    """
    annotated_sentences = []
    for doc in docs_list:
        conll_reader = ConllCorpusReader(
            path.dirname(doc),
            path.basename(doc),
            ("ignore", "ignore", "ignore", "words", "pos", "tree", "srl"),
        )
        sents_map = defaultdict(list)
        for srl_instance in conll_reader.srl_instances():
            words = [word[0] for word in srl_instance.words]
            sentence = " ".join(words)
            frame_name = srl_instance.verb_stem
            if frame_name.lower() not in valid_frames:
                continue
            trigger_locs = [
                conll_word_index_to_locs(words, index)[0] for index in srl_instance.verb
            ]

            frame_elements = []
            for argument in srl_instance.arguments:
                words_range, frame_element_name = argument
                element_start_loc = conll_word_index_to_locs(words, words_range[0])[0]
                element_end_loc = conll_word_index_to_locs(words, words_range[1] - 1)[1]
                frame_elements.append(
                    FrameElementAnnotation(
                        frame_element_name, element_start_loc, element_end_loc
                    )
                )
            sents_map[sentence].append(
                FrameAnnotation(frame_name, trigger_locs, frame_elements)
            )

        for sentence, frame_annotations in sents_map.items():
            annotated_sentences.append(
                FrameAnnotatedSentence(sentence, frame_annotations)
            )
    return annotated_sentences


class Propbank34TrainingLoader(TrainingLoader):
    """
    This loader uses ontonotes and ewt data from propbank 3.1 to train a model
    You must clone https://github.com/propbank/propbank-release and set the propbank_release_dir to the path of the cloned repo
    You must also download the LDC data for ontonotes and ewt, and run map_all_to_conll.py as described in the propbank repo
    Sadly, this data isn't free so you'll need to get it yourself before working with this loader.
    """

    propbank_release_dir: str
    train_docs: list[str] = []
    val_docs: list[str] = []
    test_docs: list[str] = []
    valid_frames: set[str] = set()

    def __init__(self, propbank_release_dir: str) -> None:
        super().__init__()
        self.propbank_release_dir = propbank_release_dir

    def setup(self) -> None:
        self.valid_frames = {frame.name.lower() for frame in load_propbank_frames()}
        self.train_docs = load_docs_set(self.propbank_release_dir, SPLITS["train"])
        self.val_docs = load_docs_set(self.propbank_release_dir, SPLITS["val"])
        self.test_docs = load_docs_set(self.propbank_release_dir, SPLITS["test"])

    def get_augmentations(self) -> list[DataAugmentation]:
        return [
            RemoveEndPunctuationAugmentation(0.5),
            SynonymAugmentation(0.3),
            KeyboardAugmentation(0.3),
            SimpleMisspellingAugmentation(0.1),
            LowercaseAugmentation(0.1),
            UppercaseAugmentation(0.1),
        ]

    def load_training_data(self) -> list[FrameAnnotatedSentence]:
        return load_propbank_samples(self.train_docs, self.valid_frames)

    def load_test_data(self) -> list[FrameAnnotatedSentence]:
        return load_propbank_samples(self.test_docs, self.valid_frames)

    def load_validation_data(self) -> list[FrameAnnotatedSentence]:
        return load_propbank_samples(self.val_docs, self.valid_frames)

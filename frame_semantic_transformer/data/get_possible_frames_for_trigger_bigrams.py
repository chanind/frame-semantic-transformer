from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
import en_core_web_sm
from nltk.stem import PorterStemmer
from .framenet import get_frames, get_lexical_units


stemmer = PorterStemmer()
nlp = en_core_web_sm.load()
MONOGRAM_BLACKLIST = {"back", "down", "make", "take", "have", "into", "come"}


def get_possible_frames_for_trigger_bigrams(bigrams: list[list[str]]) -> list[str]:
    possible_frames = []
    lookup_map = get_lexical_unit_bigram_to_frame_lookup_map()
    for bigram in bigrams:
        normalized_bigram = normalize_lexical_unit_ngram(bigram)
        if normalized_bigram in lookup_map:
            bigram_frames = lookup_map[normalized_bigram]
            possible_frames += bigram_frames
    # remove duplicates, while preserving order
    # https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set/53657523#53657523
    return list(dict.fromkeys(possible_frames))


@lru_cache(1)
def get_lexical_unit_bigram_to_frame_lookup_map() -> dict[str, list[str]]:
    uniq_lookup_map: dict[str, set[str]] = defaultdict(set)

    def add_lu_to_map(lu_name: str, frame: str) -> None:
        parts = lu_name.split()
        lu_bigrams: list[str] = []
        prev_part = None
        for part in parts:
            norm_part = normalize_lexical_unit_text(part)
            # also key this as a mongram if there's only 1 element or the word is rare enough
            if len(parts) == 1 or (
                len(norm_part) >= 4 and norm_part not in MONOGRAM_BLACKLIST
            ):
                lu_bigrams.append(normalize_lexical_unit_ngram([part]))
            if prev_part is not None:
                lu_bigrams.append(normalize_lexical_unit_ngram([prev_part, part]))
            prev_part = part

        for bigram in lu_bigrams:
            uniq_lookup_map[bigram].add(frame)

    for lu in get_lexical_units():
        add_lu_to_map(lu["name"], lu["frame"]["name"])
    for frame in get_frames():
        for lu_name in frame["lexUnit"].keys():
            add_lu_to_map(lu_name, frame["name"])
    sorted_lookup_map: dict[str, list[str]] = {}
    for lu_bigram, frames in uniq_lookup_map.items():
        sorted_lookup_map[lu_bigram] = sorted(list(frames))
    return sorted_lookup_map


def normalize_lexical_unit_ngram(ngram: list[str]) -> str:
    normalized_toks = [normalize_lexical_unit_text(tok) for tok in ngram]
    return "_".join([tok for tok in normalized_toks if tok != ""])


@lru_cache(5000)
def lemmatize_and_stem(word: str) -> str:
    if word == "":
        return word
    nlp_parse = nlp(word)
    lemma = word
    if len(nlp_parse) > 0:
        lemma = nlp_parse[0].lemma_
    return debritishize(stemmer.stem(lemma))


def debritishize(word: str) -> str:
    if len(word) <= 3:
        return word
    new_word = word.replace("our", "or").replace("aero", "air").replace("ae", "e")
    new_word = re.sub(r"ise$", "ize", new_word)
    new_word = re.sub(r"tre$", "ter", new_word)
    return new_word


CONTRACTION_NORMALIZATIONS = {
    "'ve": "have",
    "'nt": "not",
    "'ll": "will",
}

MISC_NORMALIZATIONS = {"could": "can"}


def normalize_lexical_unit_text(lu: str) -> str:
    normalized_lu = lu.lower()
    normalized_lu = re.sub(r"\.[a-zA-Z]+$", "", normalized_lu)
    normalized_lu = CONTRACTION_NORMALIZATIONS.get(normalized_lu, normalized_lu)
    normalized_lu = re.sub(r"[^a-z0-9 ]", "", normalized_lu)
    normalized_lu.strip()
    normalized_lu = lemmatize_and_stem(normalized_lu)
    MISC_NORMALIZATIONS.get(normalized_lu, normalized_lu)
    return normalized_lu

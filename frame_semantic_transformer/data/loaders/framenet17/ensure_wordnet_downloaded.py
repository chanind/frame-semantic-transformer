import nltk


def ensure_wordnet_downloaded() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

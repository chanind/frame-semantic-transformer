import nltk


def ensure_wordnet_downloaded() -> None:
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet")

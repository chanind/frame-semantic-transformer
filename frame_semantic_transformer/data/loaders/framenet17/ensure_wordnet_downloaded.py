import nltk


def ensure_wordnet_downloaded() -> None:
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4.zip")
    except LookupError:
        nltk.download("omw-1.4")

import nltk


def ensure_framenet_downloaded() -> None:
    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")

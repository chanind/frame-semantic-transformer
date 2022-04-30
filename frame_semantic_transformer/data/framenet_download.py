import nltk


def ensure_framenet_downloaded() -> None:
    nltk.download("framenet_v17")

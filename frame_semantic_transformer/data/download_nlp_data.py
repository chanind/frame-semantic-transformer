import nltk


def ensure_nlp_data_downloaded() -> None:
    for corpus in ["framenet_v17"]:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus)

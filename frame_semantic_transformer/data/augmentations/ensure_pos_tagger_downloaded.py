import nltk


def ensure_pos_tagger_downloaded() -> None:
    """
    nlpaug's SynonymAug POS-tags via nltk.pos_tag, but only ever downloads the
    tagger under its pre-3.9 name. nltk >= 3.9 looks for the language-suffixed
    "averaged_perceptron_tagger_eng" instead, so pos_tag raises LookupError
    unless we fetch it ourselves. Try the modern name first, falling back to
    the old one for nltk < 3.9.
    """
    for resource in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"taggers/{resource}")
            return
        except LookupError:
            if nltk.download(resource, quiet=True):
                return

import nltk
from nltk.downloader import Package


# NLTK only has v1.0 of PropBank, so hackily create a NLTK package and download v3.1
propbank34 = Package(
    id="propbank-frames-3.4.0",
    url="https://github.com/propbank/propbank-frames/archive/refs/tags/v3.4.0.zip",
    name="Proposition Bank Corpus 3.4",
    checksum="e563f8c9912d53ed7e709455746875e5",
    subdir="corpora",
    size=9484561,
    unzipped_size=29870379,
)


def ensure_propbank_downloaded() -> None:
    try:
        nltk.data.find("corpora/propbank-frames-3.4.0.zip")
    except LookupError:
        nltk.download(propbank34)

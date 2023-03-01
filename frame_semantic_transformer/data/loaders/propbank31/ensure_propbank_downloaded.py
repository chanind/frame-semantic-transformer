import nltk
from nltk.downloader import Package


# NLTK only has v1.0 of PropBank, so hackily create a NLTK package and download v3.1
propbank31 = Package(
    id="propbank-frames-3.1",
    url="https://github.com/propbank/propbank-frames/archive/refs/tags/v3.1.zip",
    name="Proposition Bank Corpus 3.1",
    checksum="a2474ea2fdd2599bf101661e8ccba682",
    subdir="corpora",
    size=6969233,
    unzipped_size=16889914,
)


def ensure_propbank_downloaded() -> None:
    try:
        nltk.data.find("corpora/propbank-frames-3.1.zip")
    except LookupError:
        nltk.download(propbank31)

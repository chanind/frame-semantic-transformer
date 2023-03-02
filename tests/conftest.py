from __future__ import annotations

import pytest
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.loaders.framenet17.ensure_framenet_downloaded import (
    ensure_framenet_downloaded,
)
from frame_semantic_transformer.data.loaders.propbank31.ensure_propbank_downloaded import (
    ensure_propbank_downloaded,
)

ensure_framenet_downloaded()
ensure_propbank_downloaded()

_loader_cache = LoaderDataCache(Framenet17InferenceLoader())
_training_loader = Framenet17TrainingLoader()


@pytest.fixture
def loader_cache() -> LoaderDataCache:
    return _loader_cache


@pytest.fixture
def training_loader() -> Framenet17TrainingLoader:
    return _training_loader

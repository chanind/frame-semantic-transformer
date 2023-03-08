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
from frame_semantic_transformer.data.loaders.framenet17.ensure_wordnet_downloaded import (
    ensure_wordnet_downloaded,
)

ensure_wordnet_downloaded()
ensure_framenet_downloaded()

_loader_cache = LoaderDataCache(Framenet17InferenceLoader())
# exemplars are really slow to load, so skip those for this fixture
_training_loader = Framenet17TrainingLoader(include_exemplars=False)


@pytest.fixture
def loader_cache() -> LoaderDataCache:
    return _loader_cache


@pytest.fixture
def training_loader() -> Framenet17TrainingLoader:
    return _training_loader

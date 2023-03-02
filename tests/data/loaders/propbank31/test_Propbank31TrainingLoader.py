from __future__ import annotations

import os

from syrupy import SnapshotAssertion
from frame_semantic_transformer.data.loaders.propbank31.load_propbank_frames import (
    load_propbank_frames,
)

from frame_semantic_transformer.data.loaders.propbank31.Propbank31TrainingLoader import (
    load_propbank_samples,
)

SAMPLE_CONLL_FILE = os.path.join(os.path.dirname(__file__), "samples.gold_conll")


def test_load_propbank_samples(snapshot: SnapshotAssertion) -> None:
    valid_frames = {frame.name.lower() for frame in load_propbank_frames()}
    samples = load_propbank_samples([SAMPLE_CONLL_FILE], valid_frames)
    assert len(samples) == 5
    assert samples == snapshot

from __future__ import annotations

import os

from syrupy import SnapshotAssertion

from frame_semantic_transformer.data.loaders.propbank31.Propbank31TrainingLoader import (
    load_propbank_samples,
)

SAMPLE_CONLL_FILE = os.path.join(os.path.dirname(__file__), "samples.gold_conll")


def test_load_propbank_samples(snapshot: SnapshotAssertion) -> None:
    samples = load_propbank_samples([SAMPLE_CONLL_FILE])
    assert len(samples) == 5
    assert samples == snapshot

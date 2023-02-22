from __future__ import annotations

import os


MODEL_MAX_LENGTH = 512
OFFICIAL_RELEASES = ["base", "small"]  # TODO: small, large
MODEL_REVISION = "v0.1.0"
PADDING_LABEL_ID = -100
DEFAULT_NUM_WORKERS = os.cpu_count() or 2

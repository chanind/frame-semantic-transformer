from __future__ import annotations
from typing import Any
from dataclasses import asdict

from flask import Flask, request, abort
from flask_cors import CORS
from frame_semantic_transformer import FrameSemanticTransformer

app = Flask(__name__)
CORS(app)


transformers = {
    "base": FrameSemanticTransformer("base", max_batch_size=2),
    "small": FrameSemanticTransformer("small", max_batch_size=2),
}


@app.errorhandler(400)
def handle_error(error: Any) -> Any:
    return error.description, 400


@app.route("/")
def index() -> dict[str, Any]:
    return {"name": "Frame Semantic Transformer Demo"}


@app.route("/detect-frames")
def detect_frames() -> dict[str, Any]:
    sentence = request.args.get("sentence", type=str)
    if sentence is None or len(sentence) == 0:
        abort(
            400,
            {
                "type": "invalid_params",
                "message": 'You must provide a "sentence" query param',
            },
        )
    model_size = request.args.get("model", type="str")
    transformer = transformers.get(model_size, transformers["base"])
    detect_frames_result = transformer.detect_frames(sentence)
    return asdict(detect_frames_result)

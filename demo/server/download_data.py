from frame_semantic_transformer import FrameSemanticTransformer
from frame_semantic_transformer.data.loaders.framenet17.ensure_framenet_downloaded import (
    ensure_framenet_downloaded,
)

if __name__ == "__main__":
    for model in ["base", "small"]:
        FrameSemanticTransformer(model)
    ensure_framenet_downloaded()

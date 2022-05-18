from frame_semantic_transformer import FrameSemanticTransformer
from frame_semantic_transformer.data.framenet import ensure_framenet_downloaded

if __name__ == "__main__":
    # download / use just the base model for now
    FrameSemanticTransformer("base")
    ensure_framenet_downloaded()

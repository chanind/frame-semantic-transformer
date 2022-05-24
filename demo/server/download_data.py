from frame_semantic_transformer import FrameSemanticTransformer
from frame_semantic_transformer.data.download_nlp_data import ensure_nlp_data_downloaded

if __name__ == "__main__":
    # download / use just the base model for now
    FrameSemanticTransformer("base")
    ensure_nlp_data_downloaded()

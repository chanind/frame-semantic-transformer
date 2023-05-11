# Frame Semantic Transformer

[![ci](https://img.shields.io/github/actions/workflow/status/chanind/frame-semantic-transformer/ci.yaml?branch=main)](https://github.com/chanind/frame-semantic-transformer)
[![PyPI](https://img.shields.io/pypi/v/frame-semantic-transformer?color=blue)](https://pypi.org/project/frame-semantic-transformer/)

Frame-based semantic parsing library trained on [FrameNet](https://framenet2.icsi.berkeley.edu/) and built on HuggingFace's [T5 Transformer](https://huggingface.co/docs/transformers/model_doc/t5)

**Live Demo: [chanind.github.io/frame-semantic-transformer](https://chanind.github.io/frame-semantic-transformer)**

Full docs: [frame-semantic-transformer.readthedocs.io](https://frame-semantic-transformer.readthedocs.io/)

## About

This library draws heavily on [Open-Sesame](https://github.com/swabhs/open-sesame) ([paper](https://arxiv.org/abs/1706.09528)) for inspiration on training and evaluation on FrameNet 1.7, and uses ideas from the paper [Open-Domain Frame Semantic Parsing Using Transformers](https://arxiv.org/abs/2010.10998) for using T5 as a frame-semantic parser. [SimpleT5](https://github.com/Shivanandroy/simpleT5) was also used as a base for the initial training setup.

More details: [FrameNet Parsing with Transformers Blog Post](https://chanind.github.io/ai/2022/05/24/framenet-transformers.html)

## Performance

This library uses the same train/dev/test documents and evaluation methodology as Open-Sesame, so that the results should be comparable between the 2 libraries. There are 2 pretrained models available, `base` and `small`, corresponding to `t5-base` and `t5-small` in Huggingface, respectively.

| Task                   | Sesame F1 (dev/test) | Small Model F1 (dev/test) | Base Model F1 (dev/test) |
| ---------------------- | -------------------- | ------------------------- | ------------------------ |
| Trigger identification | 0.80 / 0.73          | 0.75 / 0.71               | 0.78 / 0.74              |
| Frame classification   | 0.90 / 0.87          | 0.87 / 0.86               | 0.91 / 0.89              |
| Argument extraction    | 0.61 / 0.61          | 0.76 / 0.73               | 0.78 / 0.75              |

The base model performs similarly to Open-Sesame on trigger identification and frame classification tasks, but outperforms it by a significant margin on argument extraction. The small pretrained model has lower F1 than base across the board, but is 1/4 the size and still outperforms Open-Sesame at argument extraction.

## Installation

```
pip install frame-semantic-transformer
```

## Usage

### Inference

The main entry to interacting with the library is the `FrameSemanticTransformer` class, as shown below. For inference the `detect_frames()` method is likely all that is needed to perform frame parsing.

```python
from frame_semantic_transformer import FrameSemanticTransformer

frame_transformer = FrameSemanticTransformer()

result = frame_transformer.detect_frames("The hallway smelt of boiled cabbage and old rag mats.")

print(f"Results found in: {result.sentence}")
for frame in result.frames:
    print(f"FRAME: {frame.name}")
    for element in frame.frame_elements:
        print(f"{element.name}: {element.text}")
```

The result returned from `detect_frames()` is an object containing `sentence`, a parsed version of the original sentence text, `trigger_locations`, the indices within the sentence where frame triggers were detected, and `frames`, a list of all detected frames in the sentence. Within `frames`, each object containes `name` which corresponds to the FrameNet name of the frame, `trigger_location` corresponding to which trigger in the text this frame this frame uses, and `frame_elements` containing a list of all relevant frame elements found in the text.

For more efficient bulk processing of text, there's a `detect_frames_bulk` method which will process a list of sentences in batches. You can control the batch size using the `batch_size` param. By default this is `8`.

```python
frame_transformer = FrameSemanticTransformer(batch_size=16)

result = frame_transformer.detect_frames_bulk([
    "I'm getting quite hungry, but I can wait a bit longer.",
    "The chef gave the food to the customer.",
    "The hallway smelt of boiled cabbage and old rag mats.",
])
```

**Note**: It's not recommended to pass more than a single sentence per string to `detect_frames()` or `detect_frames_bulk()`. If you have a paragraph of text to process, it's best to split the paragraph into a list of sentences and pass the sentences as a list to `detect_frames_bulk()`. Only single sentences per string were used during training, so it's not clear how the model will handle multiple sentences in the same string.

```python
# ❌ Bad, don't do this
frame_transformer.detect_frames("Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair.")

# 👍 Do this instead
frame_transformer.detect_frames_bulk([
  "Fuzzy Wuzzy was a bear.",
  "Fuzzy Wuzzy had no hair.",
])
```

### Running on GPU vs CPU

By default, `FrameSemanticTransformer` will attempt to use a GPU if one is available. If you'd like to explictly set whether to run on GPU vs CPU, you can pass the `use_gpu` param.

```python
# force the model to run on the CPU
frame_transformer = FrameSemanticTransformer(use_gpu=False)
```

### Loading Models

There are currently 2 available pre-trained models for inference, called `base` and `small`, fine-tuned from HuggingFace's [t5-base](https://huggingface.co/t5-base) and [t5-small](https://huggingface.co/t5-small) model respectively. If a local fine-tuned t5 model exists that can be loaded as well. If no model is specified, the `base` model will be used.

```
base_transformer = FrameSemanticTransformer("base") # this is also the default
small_transformer = FrameSemanticTransformer("small") # a smaller pretrained model which is faster to run
custom_transformer = FrameSemanticTransformer("/path/to/model") # load a custom t5 model
```

By default, models are lazily loaded when `detect_frames()` is first called. If you want to load the model sooner, you can call `setup()` on a `FrameSemanticTransformer` instance to load models immediately.

```
frame_transformer = FrameSemanticTransformer()
frame_transformer.setup() # load models immediately
```

## Contributing

Any contributions to improve this project are welcome! Please open an issue or pull request in this repo with any bugfixes / changes / improvements you have!

This project uses [Black](https://github.com/psf/black) for code formatting, [Flake8](https://flake8.pycqa.org/en/latest/) for linting, and [Pytest](https://docs.pytest.org/) for tests. Make sure any changes you submit pass these code checks in your PR. If you have trouble getting these to run feel free to open a pull-request regardless and we can discuss further in the PR.

## License

The code contained in this repo is released under a MIT license, however the pretrained models are released under an Apache 2.0 license in accordance with FrameNet training data and HuggingFace's T5 base models.

## Citation

If you use Frame semantic transformer in your work, please cite the following:

```bibtex
@article{chanin2023opensource,
  title={Open-source Frame Semantic Parsing},
  author={Chanin, David},
  journal={arXiv preprint arXiv:2303.12788},
  year={2023}
}
```

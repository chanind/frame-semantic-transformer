# Frame Semantic Transformer

[![ci](https://img.shields.io/github/actions/workflow/status/chanind/frame-semantic-transformer/ci.yaml?branch=main)](https://github.com/chanind/frame-semantic-transformer)
[![PyPI](https://img.shields.io/pypi/v/frame-semantic-transformer?color=blue)](https://pypi.org/project/frame-semantic-transformer/)

Frame-based semantic parsing library trained on [FrameNet](https://framenet2.icsi.berkeley.edu/) and built on HuggingFace's [T5 Transformer](https://huggingface.co/docs/transformers/model_doc/t5)

**Live Demo: [chanind.github.io/frame-semantic-transformer](https://chanind.github.io/frame-semantic-transformer)**

## About

This library draws heavily on [Open-Sesame](https://github.com/swabhs/open-sesame) ([paper](https://arxiv.org/abs/1706.09528)) for inspiration on training and evaluation on FrameNet 1.7, and uses ideas from the paper [Open-Domain Frame Semantic Parsing Using Transformers](https://arxiv.org/abs/2010.10998) for using T5 as a frame-semantic parser. [SimpleT5](https://github.com/Shivanandroy/simpleT5) was also used as a base for the initial training setup.

More details: [FrameNet Parsing with Transformers Blog Post](https://chanind.github.io/ai/2022/05/24/framenet-transformers.html)

## Performance

This library uses the same train/dev/test documents and evaluation methodology as Open-Sesame, so that the results should be comparable between the 2 libraries. There are 2 pretrained models available, `base` and `small`, corresponding to `t5-base` and `t5-small` in Huggingface, respectively.

| Task                   | Sesame F1 (dev/test) | Small Model F1 (dev/test) | Base Model F1 (dev/test) |
| ---------------------- | -------------------- | ------------------------- | ------------------------ |
| Trigger identification | 0.80 / 0.73          | 0.74 / 0.70               | 0.78 / 0.71              |
| Frame classification   | 0.90 / 0.87          | 0.83 / 0.81               | 0.89 / 0.87              |
| Argument extraction    | 0.61 / 0.61          | 0.68 / 0.70               | 0.74 / 0.72              |

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

### Running on GPU vs CPU

By default, `FrameSemanticTransformer` will attempt to use a GPU if one is available. If you'd like to explictly set whether to run on GPU vs CPU, you can pass the `use_gpu` param.

```python
# force the model to run on the CPU
frame_transformer = FrameSemanticTransformer(use_gpu=False)
```

### Loading Models

There are currently 2 available pre-trained models for inference, called `base` and `small`, fine-tuned from HuggingFace's [t5-base](https://huggingface.co/t5-base) and [t5-small](https://huggingface.co/t5-base) model respectively. If a local fine-tuned t5 model exists that can be loaded as well. If no model is specified, the `base` model will be used.

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

### Training

If you want to train a new model on the Framenet 1.7 dataset yourself, you can run the training script like below:

```
python -m frame_semantic_transformer.train \
    --base-model t5-base \
    --use-gpu \
    --batch-size 8 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --output-dir ./outputs
```

Training uses [Pytorch Lightning](https://www.pytorchlightning.ai/) behind the scenes, and will place tensorboard logs into `./lightning_logs` as it trains.

If you need more control, you can also directly import the `train()` method from `frame_semantic_transformer.train` and run training directly in code.

### Training on different datasets

By default FrameSemanticTransformer assumes you want to train on the framenet 1.7 dataset, and will download this dataset during training and inference. If you'd like to train on a different dataset, for example a different language version of framenet, or a custom frame dataset, you'll need to provide custom data loaders to load the data for your frames. Specifically, this requires extending an instance of `TrainingLoader` to load your training data, and `InstanceLoader` to load all the frames and lexical units from your custom dataset.

These loaders have the following signatures:

```python
class InferenceLoader(ABC):
    def setup(self) -> None:
        """
        Perform any setup required, e.g. downloading needed data
        """
        pass

    @abstractmethod
    def load_frames(self) -> list[Frame]:
        """
        Load the full list of frames to be used during inference
        """
        pass

    @abstractmethod
    def normalize_lexical_unit_text(self, lu: str) -> str:
        """
        Normalize a lexical unit like "takes.v" to "take".
        """
        pass

    def prioritize_lexical_unit(self, lu: str) -> bool:
        """
        Check if the lexical unit is relatively rare, so that it should be considered "high information"
        """
        return len(self.normalize_lexical_unit_text(lu)) >= 6


class TrainingLoader(ABC):
    def setup(self) -> None:
        """
        Perform any setup required, e.g. downloading needed data.
        """
        pass

    @abstractmethod
    def get_augmentations(self) -> list[DataAugmentation]:
        """
        Get a list of augmentations to apply to the training data
        """
        pass

    @abstractmethod
    def load_training_data(self) -> list[FrameAnnotatedSentence]:
        """
        Load the training data
        """
        pass

    @abstractmethod
    def load_validation_data(self) -> list[FrameAnnotatedSentence]:
        """
        Load the validation data
        """
        pass

    @abstractmethod
    def load_test_data(self) -> list[FrameAnnotatedSentence]:
        """
        Load the test data
        """
        pass

```

The most difficult part of this is returning instances of `Frame` for the `load_frames` method of `InstanceLoader`, and `FrameAnnotatedSentence` from the `TrainingLoader`. These are simple Python dataclasses with the following signatures:

```python
@dataclass
class Frame:
    """
    Representation of a FrameNet frame
    For training on your own data, you can use this class to represent your own frames
    """

    name: str
    core_elements: list[str]
    non_core_elements: list[str]
    lexical_units: list[str]


@dataclass
class FrameAnnotatedSentence:
    """
    Representation of a sentence with annotations for use in training
    If training on your own data, you'll need to create instances of this class for your training sentences
    """

    text: str
    annotations: list[FrameAnnotation]


@dataclass
class FrameAnnotation:
    """
    A single frame occuring in a sentence
    """

    frame: str
    trigger_locs: list[int]
    frame_elements: list[FrameElementAnnotation]


@dataclass
class FrameElementAnnotation:
    """
    A single frame element in a frame annotation.
    Includes the name of the frame element and the start and end locations of the frame element in the sentence
    """

    name: str
    start_loc: int
    end_loc: int
```

Hopefully the meaning of the fields in the `Frame` dataclass should be obvious when looking at a sample [FrameNet Frame](https://framenet.icsi.berkeley.edu/fndrupal/frameIndex).

The `FrameAnnotatedSentence` class is a bit trickier, as this represents an annotated training sample. The `text` field should be a single sentence, and all `start_loc`, `end_loc`, and `trigger_locs` are indices which refer to positions in the text.

`FrameAnnotation` refers to a single frame inside of the sentence. There may be multiple frames in a sentence, which is why the `annotations` field on `FrameAnnotatedSentence` is a list of `FrameAnnotation`s. The `trigger_locs` field in `FrameAnnotation` is just the **start** locations of any triggers in the sentence for the frame. End locations of triggers are not used currently by FrameSemanticTransformer as it makes the labeling more complicated. There is an implicit assumptions here, which is that a single location in a sentence can only be a trigger for 1 frame.

`FrameElement` refers to the location of a frame element in the sentence for the frame being annotated. Frame elements do require both start and end locations in the sentence.

For instance, for the sentence "It was no use trying the lift", we have 2 frames "Attempt_means" at index 14 (the word "trying"), and "Connecting_architecture" at index 25 (the word "lift"). "Attempt_means" has a single frame element "Means" with text "the lift" (index 21 - 29), and "Connecting_architecture" likewise also has a single frame element "Part" with text "lift" (index 25 - 29). This would look like the following when turned into a `FrameAnnotatedSentence` instance:

```python
annotated_sentence = FrameAnnotatedSentence(
    text="It was no use trying the lift",
    annotations=[
        FrameAnnotation(
            frame="Attempt_means",
            trigger_locs=[14],
            frame_elements=[
                FrameElementAnnotation(
                    name="Means",
                    start_loc=21,
                    end_loc=29,
                )
            ]
        ),
        FrameAnnotation(
            frame="Connecting_architecture",
            trigger_locs=[25],
            frame_elements=[
                FrameElementAnnotation(
                    name="Part",
                    start_loc=25,
                    end_loc=29,
                )
            ]
        )
    ]
)
```

After creating custom `TrainingLoader` and `InferenceLoader` classes, you'll need to pass these classes in when training a new model and when running inference after training. An example of this is shown below:

```python
from frame_semantic_transformer import TrainingLoader, InferenceLoader, FrameSemanticTransformer
from frame_semantic_transformer.training import train

class MyCustomInferenceLoader(InferenceLoader):
    ...

class MyCustomTrainingLoader(TrainingLoader):
    ...

my_inference_loader = MyCustomInferenceLoader()
my_training_loader = MyCustomTrainingLoader()

my_model, my_tokenizer = train(
    base_model=f"t5-small",
    batch_size=32,
    max_epochs=16,
    lr=5e-5,
    inference_loader=my_inference_loader,
    training_loader=my_training_loader,
)

my_model.save_pretrained('./my_model')
my_tokenizer.save_pretrained('./my_model')

# after training...

frame_transformer = FrameSemanticTransformer('./my_model', inference_loader=my_inference_loader)
frame_transformer.detect_frames(...)
```

You can see examples of how these classes are implemented for the default framenet 1.7 by looking at [Framenet17InferenceLoader.py](https://github.com/chanind/frame-semantic-transformer/blob/main/frame_semantic_transformer/data/loaders/framenet17/Framenet17InferenceLoader.py) and [Framenet17TrainingLoader.py](https://github.com/chanind/frame-semantic-transformer/blob/main/frame_semantic_transformer/data/loaders/framenet17/Framenet17TrainingLoader.py). There's also an example of creating custom loaders for Swedish in the following Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HsntVN-YzlJxLGL0tpBaF7-4Lkvh0Bz6?usp=sharing)

If you have trouble creating and using custom loader classes please don't hesitate to [open an issue](https://github.com/chanind/frame-semantic-transformer/issues/new)!

## Contributing

Any contributions to improve this project are welcome! Please open an issue or pull request in this repo with any bugfixes / changes / improvements you have!

This project uses [Black](https://github.com/psf/black) for code formatting, [Flake8](https://flake8.pycqa.org/en/latest/) for linting, and [Pytest](https://docs.pytest.org/) for tests. Make sure any changes you submit pass these code checks in your PR. If you have trouble getting these to run feel free to open a pull-request regardless and we can discuss further in the PR.

## License

The code contained in this repo is released under a MIT license, however the pretrained models are released under an Apache 2.0 license in accordance with FrameNet training data and HuggingFace's T5 base models.

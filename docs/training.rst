Training
========

Training on Framenet 1.7
''''''''''''''''''''''''

If you want to train a new model on the Framenet 1.7 dataset yourself, you can run the training script like below:

.. code-block:: python

    python -m frame_semantic_transformer.train \
        --base-model t5-base \
        --use-gpu \
        --batch-size 8 \
        --epochs 10 \
        --learning-rate 5e-5 \
        --output-dir ./outputs

Training uses `Pytorch Lightning`_ behind the scenes, and will place tensorboard logs into `./lightning_logs` as it trains.

If you need more control, you can also directly import the `train()` method from `frame_semantic_transformer.train` and run training directly in code.

Training on custom datasets
''''''''''''''''''''''''''''''

By default FrameSemanticTransformer assumes you want to train on the framenet 1.7 dataset, and will download this dataset during training and inference. If you'd like to train on a different dataset, for example a different language version of framenet, or a custom frame dataset, you'll need to provide custom data loaders to load the data for your frames. Specifically, this requires extending an instance of `TrainingLoader` to load your training data, and `InstanceLoader` to load all the frames and lexical units from your custom dataset.

These loaders have the following signatures:

.. code-block:: python

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

The most difficult part of this is returning instances of `Frame` for the `load_frames` method of `InstanceLoader`, and `FrameAnnotatedSentence` from the `TrainingLoader`. These are simple Python dataclasses with the following signatures:

.. code-block:: python

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


Hopefully the meaning of the fields in the `Frame` dataclass should be obvious when looking at a sample `FrameNet Frame`_.

The `FrameAnnotatedSentence` class is a bit trickier, as this represents an annotated training sample. The `text` field should be a single sentence, and all `start_loc`, `end_loc`, and `trigger_locs` are indices which refer to positions in the text.

`FrameAnnotation` refers to a single frame inside of the sentence. There may be multiple frames in a sentence, which is why the `annotations` field on `FrameAnnotatedSentence` is a list of `FrameAnnotation`s. The `trigger_locs` field in `FrameAnnotation` is just the **start** locations of any triggers in the sentence for the frame. End locations of triggers are not used currently by FrameSemanticTransformer as it makes the labeling more complicated. There is an implicit assumptions here, which is that a single location in a sentence can only be a trigger for 1 frame.

`FrameElement` refers to the location of a frame element in the sentence for the frame being annotated. Frame elements do require both start and end locations in the sentence.

For instance, for the sentence "It was no use trying the lift", we have 2 frames "Attempt_means" at index 14 (the word "trying"), and "Connecting_architecture" at index 25 (the word "lift"). "Attempt_means" has a single frame element "Means" with text "the lift" (index 21 - 29), and "Connecting_architecture" likewise also has a single frame element "Part" with text "lift" (index 25 - 29). This would look like the following when turned into a `FrameAnnotatedSentence` instance:

.. code-block:: python

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

After creating custom `TrainingLoader` and `InferenceLoader` classes, you'll need to pass these classes in when training a new model and when running inference after training. An example of this is shown below:

.. code-block:: python

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

You can see examples of how these classes are implemented for the default framenet 1.7 by looking at `Framenet17InferenceLoader.py`_ and `Framenet17TrainingLoader.py`_. There's also an example of creating custom loaders for Swedish in the following Colab notebook: |Open in Colab|_

If you have trouble creating and using custom loader classes please don't hesitate to `open an issue`_!


.. _`Pytorch Lightning`: https://www.pytorchlightning.ai/
.. _`FrameNet Frame`: https://framenet.icsi.berkeley.edu/fndrupal/frameIndex
.. _`FrameNet 1.7`: https://framenet.icsi.berkeley.edu/fndrupal/
.. _`Framenet17InferenceLoader.py`: https://github.com/chanind/frame-semantic-transformer/blob/main/frame_semantic_transformer/data/loaders/framenet17/Framenet17InferenceLoader.py
.. _`Framenet17TrainingLoader.py`: https://github.com/chanind/frame-semantic-transformer/blob/main/frame_semantic_transformer/data/loaders/framenet17/Framenet17TrainingLoader.py
.. _`open an issue`: https://github.com/chanind/frame-semantic-transformer/issues/new

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _Open in Colab: https://colab.research.google.com/drive/1HsntVN-YzlJxLGL0tpBaF7-4Lkvh0Bz6?usp=sharing
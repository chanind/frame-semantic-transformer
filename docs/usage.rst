Usage
=====

Inference
'''''''''

The main entry to interacting with the library is the `FrameSemanticTransformer` class, as shown below. For inference the `detect_frames()` method is likely all that is needed to perform frame parsing.

.. code-block:: python

    from frame_semantic_transformer import FrameSemanticTransformer

    frame_transformer = FrameSemanticTransformer()

    result = frame_transformer.detect_frames("The hallway smelt of boiled cabbage and old rag mats.")

    print(f"Results found in: {result.sentence}")
    for frame in result.frames:
        print(f"FRAME: {frame.name}")
        for element in frame.frame_elements:
            print(f"{element.name}: {element.text}")

The result returned from `detect_frames()` is an object containing `sentence`, a parsed version of the original sentence text, `trigger_locations`, the indices within the sentence where frame triggers were detected, and `frames`, a list of all detected frames in the sentence. Within `frames`, each object containes `name` which corresponds to the FrameNet name of the frame, `trigger_location` corresponding to which trigger in the text this frame this frame uses, and `frame_elements` containing a list of all relevant frame elements found in the text.

Bulk inference
''''''''''''''

For more efficient bulk processing of text, there's a `detect_frames_bulk` method which will process a list of sentences in batches. You can control the batch size using the `batch_size` param. By default this is `8`.

.. code-block:: python

    frame_transformer = FrameSemanticTransformer(batch_size=16)

    result = frame_transformer.detect_frames_bulk([
        "I'm getting quite hungry, but I can wait a bit longer.",
        "The chef gave the food to the customer.",
        "The hallway smelt of boiled cabbage and old rag mats.",
    ])


**Note**: It's not recommended to pass more than a single sentence per string to `detect_frames()` or `detect_frames_bulk()`. If you have a paragraph of text to process, it's best to split the paragraph into a list of sentences and pass the sentences as a list to `detect_frames_bulk()`. Only single sentences per string were used during training, so it's not clear how the model will handle multiple sentences in the same string.

.. code-block:: python

    # ❌ Bad, don't do this
    frame_transformer.detect_frames("Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair.")

    # 👍 Do this instead
    frame_transformer.detect_frames_bulk([
        "Fuzzy Wuzzy was a bear.",
        "Fuzzy Wuzzy had no hair.",
    ])

Running on GPU vs CPU
''''''''''''''''''''''

By default, `FrameSemanticTransformer` will attempt to use a GPU if one is available. If you'd like to explictly set whether to run on GPU vs CPU, you can pass the `use_gpu` param.

.. code-block:: python

    # force the model to run on the CPU
    frame_transformer = FrameSemanticTransformer(use_gpu=False)

Loading models
''''''''''''''

There are currently 2 available pre-trained models for inference, called `base` and `small`, fine-tuned from HuggingFace's `t5-base`_ and `t5-small`_ model respectively. If a local fine-tuned t5 model exists that can be loaded as well. If no model is specified, the `base` model will be used.

.. code-block:: python

    base_transformer = FrameSemanticTransformer("base") # this is also the default
    small_transformer = FrameSemanticTransformer("small") # a smaller pretrained model which is faster to run
    custom_transformer = FrameSemanticTransformer("/path/to/model") # load a custom t5 model

By default, models are lazily loaded when `detect_frames()` is first called. If you want to load the model sooner, you can call `setup()` on a `FrameSemanticTransformer` instance to load models immediately.

.. code-block:: python

    frame_transformer = FrameSemanticTransformer()
    frame_transformer.setup() # load models immediately

.. _t5-base: https://huggingface.co/t5-base
.. _t5-small: https://huggingface.co/t5-small
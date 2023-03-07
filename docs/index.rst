Frame Semantic Transformer
=================================================

Frame-based semantic parsing library trained on `FrameNet 1.7`_ and built on HuggingFace's `T5 Transformer`_

.. image:: https://img.shields.io/pypi/v/frame-semantic-transformer.svg?color=blue
   :target: https://pypi.org/project/frame-semantic-transformer
   :alt: PyPI

.. image:: https://img.shields.io/github/actions/workflow/status/chanind/frame-semantic-transformer/ci.yaml?branch=main
   :target: https://github.com/chanind/frame-semantic-transformer
   :alt: Build Status


**Live Demo:** `chanind.github.io/frame-semantic-transformer <https://chanind.github.io/frame-semantic-transformer>`_

Installation
------------
Frame Semantic Transformer releases are hosted on `PyPI`_, and can be installed using `pip` as below:

.. code-block:: bash

   pip install frame-semantic-transformer

Basic usage
-----------

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


.. toctree::
   :maxdepth: 2

   usage
   training
   about

.. toctree::
   :caption: Project Links

   GitHub <https://github.com/chanind/frame-semantic-transformer>
   PyPI <https://pypi.org/project/frame-semantic-transformer>



.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


.. _PyPI: https://pypi.org/project/frame-semantic-transformer/
.. _FrameNet 1.7: https://framenet2.icsi.berkeley.edu/
.. _T5 Transformer: https://huggingface.co/docs/transformers/model_doc/t5
.. _Open-Sesame: https://github.com/swabhs/open-sesame
.. _SimpleT5: https://github.com/Shivanandroy/simpleT5
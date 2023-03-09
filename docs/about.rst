About
=====

This library draws heavily on `Open-Sesame`_ (`paper <https://arxiv.org/abs/1706.09528>`_) for inspiration on training and evaluation on FrameNet 1.7, and uses ideas from the paper `Open-Domain Frame Semantic Parsing Using Transformers <https://arxiv.org/abs/2010.10998>`_ for using T5 as a frame-semantic parser. `SimpleT5`_ was also used as a base for the initial training setup.

More details: `FrameNet Parsing with Transformers Blog Post <https://chanind.github.io/ai/2022/05/24/framenet-transformers.html>`_

Performance
-----------

This library uses the same train/dev/test documents and evaluation methodology as Open-Sesame, so that the results should be comparable between the 2 libraries. There are 2 pretrained models available, `base` and `small`, corresponding to `t5-base` and `t5-small` in Huggingface, respectively.

+------------------------+----------------------+---------------------------+--------------------------+
| Task                   | Sesame F1 (dev/test) | Small Model F1 (dev/test) | Base Model F1 (dev/test) |
+========================+======================+===========================+==========================+
| Trigger identification | 0.80 / 0.73          | 0.74 / 0.70               | 0.78 / 0.71              |
+------------------------+----------------------+---------------------------+--------------------------+
| Frame classification   | 0.90 / 0.87          | 0.83 / 0.81               | 0.89 / 0.87              |
+------------------------+----------------------+---------------------------+--------------------------+
| Argument extraction    | 0.61 / 0.61          | 0.68 / 0.70               | 0.74 / 0.72              |
+------------------------+----------------------+---------------------------+--------------------------+

The base model performs similarly to Open-Sesame on trigger identification and frame classification tasks, but outperforms it by a significant margin on argument extraction. The small pretrained model has lower F1 than base across the board, but is 1/4 the size and still outperforms Open-Sesame at argument extraction.


License
-------

The Frame Semantic Transformer code is released under a MIT license, however the pretrained models are released under an Apache 2.0 license in accordance with FrameNet training data and HuggingFace's T5 base models.


.. _Open-Sesame: https://github.com/swabhs/open-sesame
.. _SimpleT5: https://github.com/Shivanandroy/simpleT5
from __future__ import annotations
from .DataAugmentation import DataAugmentation


class LowercaseAugmentation(DataAugmentation):
    def apply_augmentation(self, input: str, output: str) -> tuple[str, str]:
        task_def_index = input.find(":")
        task_def = input[:task_def_index]
        input_contents = input[task_def_index:]
        # only lowercase the content, not the task definition
        return (
            task_def + input_contents.lower(),
            output.lower(),
        )

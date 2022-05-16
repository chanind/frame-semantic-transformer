from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Sequence

from frame_semantic_transformer.data.data_utils import standardize_punct

from .TaskSample import TaskSample
from .TriggerIdentificationTask import TriggerIdentificationTask


@dataclass
class TriggerIdentificationSample(TaskSample):
    task: TriggerIdentificationTask
    trigger_locs: list[int]

    # -- input / target for training --

    def get_target(self) -> str:
        output = ""
        prev_trigger_loc = 0
        for loc in sorted(self.trigger_locs):
            # TODO: handle these special chars better
            output += self.task.text[prev_trigger_loc:loc] + "*"
            prev_trigger_loc = loc
        output += self.task.text[prev_trigger_loc:]
        return standardize_punct(output)

    @staticmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str], target: str, _input: str
    ) -> tuple[int, int, int]:
        true_pos = 0
        false_pos = 0
        false_neg = 0

        prediction_parts = process_text_for_evaluation(prediction_outputs[0]).split()
        target_parts = process_text_for_evaluation(target).split()

        for i, target_part in enumerate(target_parts):
            pred_part = "" if i >= len(prediction_parts) else prediction_parts[i]
            is_target_trigger = target_part[0] == "*"
            is_pred_trigger = pred_part != "" and pred_part[0] == "*"
            target_content = target_part.replace("*", "")
            pred_content = pred_part.replace("*", "")
            # if the prediction is too short or the text doesn't match, the item is just wrong
            if target_content != pred_content:
                if is_target_trigger:
                    false_neg += 1
                else:
                    false_pos += 1
            elif is_target_trigger and is_pred_trigger:
                true_pos += 1
            elif is_target_trigger and not is_pred_trigger:
                false_neg += 1
            elif is_pred_trigger and not is_target_trigger:
                false_pos += 1

        # if the predictions are longer than the targets, then every extra item is just wrong
        if len(prediction_parts) > len(target_parts):
            false_pos += len(prediction_parts) - len(target_parts)

        return (true_pos, false_pos, false_neg)


def process_text_for_evaluation(sent: str) -> str:
    updated_sent = standardize_punct(sent)
    updated_sent = re.sub(r"\*\s+([a-zA-Z0-9])", r"*\1", updated_sent)
    updated_sent = re.sub(r"([a-zA-Z0-9])(\*?')", r"\1 \2", updated_sent)
    return updated_sent

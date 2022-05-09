from __future__ import annotations
from dataclasses import dataclass

from frame_semantic_transformer.data.task_samples.TaskSample import TaskSample


@dataclass
class TriggerIdentificationSample(TaskSample):
    text: str
    trigger_locs: list[int]

    # -- input / target for training --

    def get_task_name(self) -> str:
        return "trigger_identification"

    def get_input(self) -> str:
        return f"TRIGGER: {self.text}"

    def get_target(self) -> str:
        output = ""
        prev_trigger_loc = 0
        for loc in sorted(self.trigger_locs):
            # TODO: handle these special chars better
            output += self.text[prev_trigger_loc:loc] + "*"
            prev_trigger_loc = loc
        output += self.text[prev_trigger_loc:]
        return output

    def evaluate_prediction(self, prediction: str) -> tuple[int, int, int]:
        true_pos = 0
        false_pos = 0
        false_neg = 0

        prediction_parts = prediction.split()
        target_parts = self.get_target().split()

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

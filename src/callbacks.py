from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class ModelInfoCallback(TrainerCallback):
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        model.print_trainable_parameters()

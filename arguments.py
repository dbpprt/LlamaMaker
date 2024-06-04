import datetime
from dataclasses import dataclass, field
from typing import Optional

from src.constants import SUPPORTED_MODELS


@dataclass
class ScriptArguments:
    experiment_name: Optional[str] = field(
        default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        metadata={
            "help": "The name of the experiment. This will be used as a folder name for all artificats of the training including tensorboard logs, checkpoints, etc."
        },
    )
    data_config: Optional[str] = field(
        default="./data/examples/llama3.yaml",
        metadata={"help": "The path to the data configuration file (see documentation for more details)."},
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "Start training in debug mode (subsample dataset, etc)."},
    )
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run eval."},
    )
    model_id: Optional[str] = field(
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": f"The model that you want to train from the Hugging Face hub. Currently tested and supported are: {','.join(SUPPORTED_MODELS)}"
        },
    )
    use_unslooth: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use unslooth library. Note: it needs to be installed separately and only supports a single NVIDIA GPU."
        },
    )
    use_4bit_training: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use 4bit training. Note: this requires a CUDA device to be available and doesn't work on MPS or CPU."
        },
    )
    use_4bit_double_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Use 4bit double quant."},
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2.5e-5)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.0)
    lora_r: Optional[int] = field(default=8)
    use_rs_lora: Optional[bool] = field(default=False)
    lora_target_modules: Optional[str] = field(default="all-linear")
    lora_modules_to_save: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=512)
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_8bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    neftune_noise_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "Neftune noise alpha."},
    )
    num_train_epochs: int = field(default=10.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    eval_steps: int = field(default=10, metadata={"help": "Run eval every X updates steps."})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    save_limit: int = field(default=3, metadata={"help": "Save limit."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./runs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

import os
import re

import torch
from datasets import load_dataset, set_caching_enabled
from omegaconf import OmegaConf
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils import is_flash_attn_2_available
from trl import SFTTrainer

from accelerate import PartialState
from arguments import ScriptArguments
from src.callbacks import ModelInfoCallback
from src.formatting import formatting_func
from src.utils import instantiate


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    data_config = OmegaConf.load(script_args.data_config)
    output_dir = os.path.join(script_args.output_dir, script_args.experiment_name)
    set_caching_enabled(False)

    trainer_args = {}

    # we only use flash attention if its available and we have cuda available
    use_flash_attention = script_args.use_flash_attention_2
    try:
        if use_flash_attention and not is_flash_attn_2_available():
            print("Flash attention 2 not available, disabling it")
            use_flash_attention = False
    except:  # noqa: E722
        # this shouldn't happen actually
        use_flash_attention = False

    lora_target_modules = (
        [x.strip() for x in script_args.lora_target_modules.split(",")]
        if script_args.lora_target_modules != "all-linear"
        else "all-linear"
    )
    lora_modules_to_save = (
        [x.strip() for x in script_args.lora_modules_to_save.split(",")] if script_args.lora_modules_to_save else None
    )

    model_kwargs = {}

    if script_args.use_4bit_training and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            # TODO: Why does it use less memory if disabled?
            bnb_4bit_use_double_quant=script_args.use_4bit_double_quant,
        )
        model_kwargs = {"quantization_config": quantization_config}
    elif script_args.use_4bit_training and not torch.cuda.is_available():
        print("4bit training is only supported on GPU, using fp/bf16 instead.")

    if torch.cuda.is_available():
        device_map = {"": PartialState().process_index}
    elif torch.backends.mps.is_available():
        print("Using mps, CUDA is not available")
        device_map = {"": "mps"}
    else:
        print("Using cpu, CUDA is not available")
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=script_args.model_id,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa" if not use_flash_attention else "flash_attention_2",
        use_cache=not script_args.gradient_checkpointing,
        **model_kwargs,
    )

    if script_args.use_4bit_training and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=script_args.lora_r,
        modules_to_save=lora_modules_to_save,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # TODO: not supported in the version of peft available on SageMaker
        # use_rslora=script_args.use_rs_lora,
    )
    trainer_args = {**trainer_args, "peft_config": lora_config}

    # TODO: is this correct?
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_id,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
    )

    # TODO: this feels so wrong...
    if re.search("tinyllama", script_args.model_id, re.IGNORECASE):
        # tokenizer.eos_token_id = 32002
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif re.search("llama", script_args.model_id, re.IGNORECASE):
        # tokenizer.eos_token_id = tokenizer.pad_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError("Model not supported")

    model.config.pad_token_id = tokenizer.pad_token_id

    # if re.match("[^/]+/llama", base_model_name):
    #     model.config.pad_token_id = tokenizer.pad_token_id = 0
    #     model.config.bos_token_id = tokenizer.bos_token_id = 1
    #     model.config.eos_token_id = tokenizer.eos_token_id = 2

    # if re.match("[^/]+/llama", base_model_name):
    #     model.config.pad_token_id = get_tokenizer(
    #         base_model_name).pad_token_id = 0
    #     model.config.bos_token_id = 1
    #     model.config.eos_token_id = 2

    collator = instantiate(data_config.collator, tokenizer=tokenizer)

    dataset = load_dataset(
        data_config.dataset.type,
        data_files={"train": data_config.dataset.train, "eval": data_config.dataset.eval},
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    if script_args.debug:
        print("Debug mode, subsampling dataset to 10 samples.")
        num_samples_to_select = 10

        train_dataset = train_dataset.select(range(num_samples_to_select))
        eval_dataset = eval_dataset.select(range(num_samples_to_select))

    training_arguments = TrainingArguments(
        report_to="tensorboard",
        do_train=script_args.do_train,
        do_eval=script_args.do_eval,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=output_dir,
        logging_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        # TODO: should this be enabled by default?
        # auto_find_batch_size=True,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        max_grad_norm=script_args.max_grad_norm,
        num_train_epochs=script_args.num_train_epochs,
        warmup_ratio=script_args.warmup_ratio,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_checkpointing=script_args.gradient_checkpointing,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing_kwargs={"use_reentrant": False} if script_args.gradient_checkpointing else None,
        neftune_noise_alpha=script_args.neftune_noise_alpha,
        load_best_model_at_end=True,
        save_total_limit=script_args.save_limit,
        # TODO: Slows down training
        # skip_memory_metrics=False,
    )

    trainer = SFTTrainer(
        callbacks=[ModelInfoCallback()],
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=script_args.packing,
        tokenizer=tokenizer,
        max_seq_length=script_args.max_seq_length,
        formatting_func=formatting_func(
            prompt=data_config.prompt,
            eos_token=tokenizer.eos_token if data_config.append_eos_token else None,
            json_fields=data_config.json_fields,
        ),
        data_collator=collator,
        # compute_metrics=compute_metrics(model, tokenizer),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **trainer_args,
    )

    trainer.train()

    if trainer.accelerator.is_main_process:
        trainer.save_model(output_dir)
        final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
        final_merged_dir = os.path.join(output_dir)  # , "final_merged_checkpoint")

        trainer.model.save_pretrained(final_checkpoint_dir)

        # free memory for merging weights
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            device_map="auto",
            offload_folder=os.path.join(output_dir, "offload"),
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        # are we merging 4bit to 16bit here?
        model = model.merge_and_unload()
        model.save_pretrained(final_merged_dir, safe_serialization=True)


if __name__ == "__main__":
    main()

import json
from enum import Enum, Flag
from typing import Any, Optional

import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from pathlib import Path
from utils.config_dataclasses import ModelConfig
import gc
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def cleanup_memory(*objects) -> None:
    """Free GPU and CPU memory."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logging.info("Memory cleaned up.")


class EnumEncoder(json.JSONEncoder):
    """JSON encoder that handles Enum and Flag types."""

    def default(self, obj):
        if isinstance(obj, (Enum, Flag)):
            return obj.name  # or obj.value
        return super().default(obj)


def fine_tune(
    model_name: str,
    model_config: ModelConfig,
    train_dataset: Any,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 4,
    wandb_run_name: Optional[str] = None,
    from_scratch: bool = True,
) -> Path:
    """Fine-tune model with LoRA and SFT, returning saved model path."""

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        if model_config.huggingface_path == "HuggingFaceTB/SmolLM2-135M"
        else [
            "Wqkv",
            "fc1",
            "fc2",
        ],
    )

    if from_scratch and model_name != model_config.huggingface_path:
        logging.warning("'from_scratch' is selected but given a local checkpoint")

    device = 'cuda' if model_config.huggingface_path == 'HuggingFaceTB/SmolLM2-135M' else None
    device_map = 'auto' if model_config.huggingface_path == 'microsoft/phi-2' else None


    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            device=device,
            trust_remote_code=True,
        )
        if from_scratch
        else AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            device=device,
            trust_remote_code=True,
            is_trainable=True,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", padding=True, truncation=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Compute steps
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    logging.info(
        f"Training plan: {steps_per_epoch} steps/epoch, total {total_steps} steps"
    )

    # SFT training configuration
    training_args = SFTConfig(
        completion_only_loss=False,
        dataloader_persistent_workers=True,
        dataloader_num_workers=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        logging_first_step=True,
        load_best_model_at_end=False,
        num_train_epochs=num_epochs,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        packing=False,
        run_name=wandb_run_name,
        report_to="wandb",
        save_strategy="epoch",
        fp16=True,
        warmup_steps=2,
        # torch_compile=True
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config if from_scratch else None,
        args=training_args,
    )

    # Start training
    logging.info("=" * 50)
    logging.info("Starting training...")
    logging.info("=" * 50)

    trainer.train()

    # Save final model
    final_model_path = f"{output_dir}/final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(
        f"Training complete! Model and tokenizer are saved to {final_model_path}"
    )
    cleanup_memory(model, tokenizer, trainer)
    return final_model_path

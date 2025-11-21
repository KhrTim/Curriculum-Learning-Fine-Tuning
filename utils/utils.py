import json
from enum import Enum, Flag
from typing import Any, Callable, Optional, Tuple

import torch
from peft import LoraConfig
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



def cleanup_memory(*objects):
        """
        Free GPU + CPU memory
        """
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
    from_scratch=True,
) -> Path:
    """
    Generic training function supporting all strategies with LoRA and SFT.

    Args:
        model_name: HuggingFace model name or path
        train_dataset: Training dataset
        output_dir: Directory to save model
        num_epochs: Number of epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        formatting_func: Function to format examples (optional)
        resume_from_checkpoint: Checkpoint path to resume training
        gradient_accumulation_steps: Steps for gradient accumulation
        wandb_run_name: Name of WandB run

    Returns:
        trainer, model, tokenizer, total_steps
    """

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", padding=True, truncation=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Compute steps
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    print(f"Training plan: {steps_per_epoch} steps/epoch, total {total_steps} steps")

    # SFT training configuration
    training_args = SFTConfig(
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
        torch_compile=True
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
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    # Save final model
    final_model_path = f"{output_dir}/final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Training complete! Model and tokenizer are saved to {final_model_path}")
    cleanup_memory(model, tokenizer, trainer)
    return final_model_path

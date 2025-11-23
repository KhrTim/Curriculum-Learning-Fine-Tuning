import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, pipeline
from peft import LoraConfig
from utils.utils import cleanup_memory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


class Evaluation:
    """Class for evaluating text-generation models on datasets like GSM8K."""

    def __init__(self):
        """Initialize evaluation class."""
        pass

    @staticmethod
    def extract_answer(text: Optional[str]) -> Optional[str]:
        """Extract numerical answer from GSM8K-format text."""
        if not text:
            return None

        # Try GSM8K style: #### [number]
        match = re.search(r"####\s*(-?\d+\.?\d*)", text)
        if match:
            return match.group(1).rstrip(".").strip()

        # Fallback: last number in text
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1].rstrip(".")
        return None

    def evaluate_model(
        self,
        model_checkpoint: Path,
        test_dataset: List[Dict[str, str]],
        batch_size: int = 8,
        max_new_tokens: int = 512,
        save_samples_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset and return metrics."""
        results = {
            "exact_match": 0,
            "total": 0,
            "correct_formatting": 0,
            "examples": [],
            "exact_match_samples": [],
        }

        # TODO: which tokenizer is loaded?
        # TODO: protect from repetition
        logging.info("Loading the model...")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        config = LoraConfig.from_pretrained(model_checkpoint)
        device = 'cuda' if config.base_model_name_or_path == 'HuggingFaceTB/SmolLM2-135M' else None
        device_map = 'auto' if config.base_model_name_or_path == 'microsoft/phi-2' else None

        pipe = pipeline(
            "text-generation",
            model=model_checkpoint,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic generation
            num_workers=16,
            device_map=device_map,
            device=device,
            dtype=torch.float16,
            no_repeat_ngram_size=3,
            encoder_repetition_penalty=1.1,
            length_penalty=0.7,
        )
        # Prepare prompts
        prompts = [f"Question: {ex['question']}\nAnswer:" for ex in test_dataset]

        logging.info("Evaluating...")

        outputs = pipe(prompts, batch_size=batch_size)

        logging.info("Calculating matches...")

        for j, output_list in enumerate(outputs):
            example = test_dataset[j]
            true_answer = self.extract_answer(example.get("answer", ""))
            generated_text = output_list[0]["generated_text"] if output_list else ""
            pred_answer = self.extract_answer(generated_text)

            exact_match = (
                (pred_answer == true_answer) if pred_answer and true_answer else False
            )

            # Check if generated text has correct GSM8K formatting (#### before answer)
            has_correct_formatting = bool(re.search(r"####\s*-?\d+\.?\d*", generated_text))

            results["exact_match"] += int(exact_match)
            results["correct_formatting"] += int(has_correct_formatting)

            # Store first few examples
            if len(results["examples"]) < 5:
                results["examples"].append(
                    {
                        "question": example["question"],
                        "true_answer": true_answer,
                        "generated": generated_text,
                        "pred_answer": pred_answer,
                        "correct": exact_match,
                    }
                )

            # Collect all exact match samples
            if exact_match:
                results["exact_match_samples"].append(
                    {
                        "question": example["question"],
                        "true_answer": true_answer,
                        "generated": generated_text,
                        "pred_answer": pred_answer,
                    }
                )

        # Compute percentages
        total = results["total"] = len(outputs)
        for metric in ["exact_match", "correct_formatting"]:
            results[f"{metric}_pct"] = (results[metric] / total * 100) if total > 0 else 0.0

        logging.info(
            f"Evaluation finished. Exact match: {results['exact_match_pct']:.2f}% | "
            f"Correct formatting: {results['correct_formatting_pct']:.2f}%"
        )

        # Save exact match samples if directory is provided
        if save_samples_dir:
            save_samples_dir.mkdir(parents=True, exist_ok=True)
            import json
            checkpoint_name = model_checkpoint.name if isinstance(model_checkpoint, Path) else str(model_checkpoint).split('/')[-3]
            samples_file = save_samples_dir / f"exact_match_samples_{checkpoint_name}.json"
            with open(samples_file, "w") as f:
                json.dump(results["exact_match_samples"], f, indent=2)
            logging.info(f"Saved {len(results['exact_match_samples'])} exact match samples to {samples_file}")

        logging.info("Cleaning up...")

        cleanup_memory(tokenizer, pipe)

        return results

import gc
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, pipeline
from utils.utils import cleanup_memory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


class Evaluation:
    """Class for evaluating text-generation models on datasets like GSM8K."""

    def __init__(self):
        """
        Evaluation class for model testing.
        Note: Models loaded with accelerate (device_map="auto") handle device placement automatically.
        """
        pass

    @staticmethod
    def extract_answer(text: Optional[str]) -> Optional[str]:
        """
        Extract final numerical answer from GSM8K-format text.

        Args:
            text: str or None

        Returns:
            Extracted answer as string, or None if not found.
        """
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
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.

        Args:
            model: Pretrained HuggingFace model or path
            test_dataset: List of dicts with 'question' and 'answer'
            tokenizer: Optional tokenizer
            batch_size: Number of prompts per batch
            max_new_tokens: Maximum tokens to generate per prompt

        Returns:
            Dict with evaluation metrics and sample outputs
        """
        results = {
            "exact_match": 0,
            "total": 0,
            "examples": [],
        }

        # TODO: which tokenizer is loaded?
        # TODO: protect from repetition
        logging.info("Loading the model...")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        pipe = pipeline(
            "text-generation",
            model=model_checkpoint,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic generation
            num_workers=16,
            device="auto",
            dtype=torch.float16,
            no_repeat_ngram_size=3,
            encoder_repetition_penalty=1.1,
            length_penalty=0.7
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

            results["exact_match"] += int(exact_match)
            results["total"] += 1

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

        # Compute percentage
        results["exact_match_pct"] = (
            (results["exact_match"] / results["total"]) * 100
            if results["total"] > 0
            else 0.0
        )

        logging.info(
            f"Evaluation has been finished. Exact match : {results['exact_match_pct']}% "
        )
        logging.info("Cleaning up...")

        cleanup_memory(tokenizer,pipe)

        return results

#!/usr/bin/env python3
"""
VASP Convergence Prediction Evaluation - Failed Sample Rerun Script

This script reads an existing evaluation results CSV file, identifies failed samples,
and reruns LLM evaluation only on those failed samples, saving new results to a supplementary file.

Example:
python scripts/rerun_failed_evaluate_llm_convergence.py \
    06_added_data/glm-4.6.csv \
    --dataset original_dataset/benchmark_dataset.csv \
    --output 06_added_data/glm-4.6_add.csv \
    --provider zhipuai \
    --model glm-4.6 \
    --steps 50 \
    --prompt-template prompt_templates/few_shot_2.prompt.md
"""

import os
import sys
import pandas as pd
import argparse
import time
from typing import Optional

# --- CORRECTED IMPORT ---
# Import core class from main evaluation script evaluate_llm_convergence.py
from evaluate_llm_convergence import LLMConvergenceEvaluator
# --- END CORRECTION ---


def rerun_failed_samples(
    results_file: str,
    original_dataset_file: str,
    output_file: Optional[str] = None,
    **kwargs,
):
    """
    Rerun failed samples.

    Args:
        results_file (str): Path to evaluation results CSV file containing failed samples.
        original_dataset_file (str): Path to original complete dataset CSV file.
        output_file (str, optional): Output filename. If None, automatically generated.
        **kwargs: Other parameters passed to LLMConvergenceEvaluator and run_evaluation.
    """
    # --- Step 1 & 2: Read results file and retrieve failed examples ---
    print(f"[*] Reading evaluation results from '{results_file}'...")
    try:
        df_results = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"[ERROR] Results file does not exist: {results_file}")
        sys.exit(1)

    # Filter failed samples (success column is False or prediction column is empty)
    # Add check for prediction column to improve robustness
    df_failed = df_results[
        (df_results["success"] == False) | (df_results["prediction"].isna())
    ]

    if df_failed.empty:
        print(
            "[*] Congratulations! No failed samples found in results file. No rerun needed."
        )
        return

    # --- Step 3: Lock failed example paths ---
    failed_filepaths = df_failed["filepath"].tolist()
    print(f"[*] Found {len(failed_filepaths)} failed samples, preparing to rerun:")
    for path in failed_filepaths:
        print(f"  - {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")

    # Extract information for these failed samples from original dataset to create a temporary "todo" dataset
    print(f"[*] Loading original dataset from '{original_dataset_file}'...")
    try:
        df_original_dataset = pd.read_csv(original_dataset_file)
    except FileNotFoundError:
        print(f"[ERROR] Original dataset file does not exist: {original_dataset_file}")
        sys.exit(1)

    df_todo = df_original_dataset[
        df_original_dataset["filepath"].isin(failed_filepaths)
    ]

    # Save this temporary DataFrame to a temporary file so run_evaluation can use it
    temp_dataset_path = f"temp_rerun_dataset_{int(time.time())}.csv"
    df_todo.to_csv(temp_dataset_path, index=False)
    print(f"[*] Created temporary task list: '{temp_dataset_path}'")

    # --- Step 4: Rerun and generate results to new file ---
    # Determine output filename
    if output_file is None:
        base, ext = os.path.splitext(results_file)
        output_file = f"{base}_add{ext}"

    print(
        f"[*] Preparing to start rerun evaluation... Output will be saved to '{output_file}'"
    )

    # Extract parameters needed by evaluator and run_evaluation from kwargs
    evaluator_args = {
        "provider": kwargs.get("provider"),
        "model": kwargs.get("model"),
        "api_key": kwargs.get("api_key"),
        "debug": kwargs.get("debug", False),
        "timeout": kwargs.get("timeout", 60),
        "max_retries": kwargs.get("max_retries", 3),
        "prompt_template": kwargs.get("prompt_template"),
    }

    run_args = {
        "n_steps": kwargs.get("steps"),
        "max_samples": None,  # Rerun all failed samples
        "call_interval": kwargs.get("call_interval", 0.5),
    }

    try:
        # Initialize evaluator
        evaluator = LLMConvergenceEvaluator(**evaluator_args)

        # Run evaluation
        evaluator.run_evaluation(
            dataset_path=temp_dataset_path, output_file=output_file, **run_args
        )
        print(f"\n[*] Rerun completed! Supplementary results saved to: {output_file}")

    except Exception as e:
        print(f"[ERROR] Error occurred during rerun: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)
            print(f"[*] Cleaned up temporary file: '{temp_dataset_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Rerun failed samples in LLM VASP evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core parameters ---
    parser.add_argument(
        "results_file",
        help="Path to evaluation results CSV file to check (e.g., 10steps.csv).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="benchmark_dataset.csv",
        help="Path to original complete dataset file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path for supplementary results. If not specified, automatically named as [original_filename]_add.csv.",
    )

    # --- LLM and evaluation parameters (must match original script) ---
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        required=True,
        help="Number of steps to truncate (must match original evaluation).",
    )
    parser.add_argument(
        "--provider",
        default="zhipuai",
        choices=["zhipuai", "siliconflow", "openrouter", "litellm"],
        help="LLM provider.",
    )
    parser.add_argument(
        "--model", required=True, help="Model name to use (required parameter)."
    )
    parser.add_argument(
        "--api-key",
        help="API key (if not specified, uses default or environment variable).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show detailed output.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="API call timeout (seconds)."
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum API call retry attempts."
    )
    parser.add_argument(
        "--call-interval",
        type=float,
        default=0.5,
        help="API call interval time (seconds).",
    )
    parser.add_argument("--prompt-template", help="External prompt template file path.")

    args = parser.parse_args()

    # Package all parameters and pass to main function
    rerun_failed_samples(
        results_file=args.results_file,
        original_dataset_file=args.dataset,
        output_file=args.output,
        # Other parameters
        steps=args.steps,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        debug=args.debug,
        timeout=args.timeout,
        max_retries=args.max_retries,
        call_interval=args.call_interval,
        prompt_template=args.prompt_template,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LLM VASP Convergence Prediction Evaluation Script

Predicts whether a calculation task will ultimately converge by analyzing the first N steps of OSZICAR file data.
Supports multiple LLM providers: ZhipuAI, SiliconFlow, OpenRouter, DeepSeek
"""

import os
import sys
import json
import re
import pandas as pd
import time
from typing import Tuple, Optional, List, Dict, Any
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_providers import create_llm_provider


class LLMConvergenceEvaluator:
    """LLM convergence prediction evaluator"""

    def __init__(
        self,
        provider: str = "zhipuai",
        model: str = None,
        api_key: str = None,
        debug: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        prompt_template: str = None,
    ):
        """
        Initialize evaluator

        Args:
            provider: LLM provider name (zhipuai, siliconflow, openrouter, litellm, deepseek)
            model: Model name to use
            api_key: API key
            debug: Whether to enable debug mode
            timeout: API call timeout (seconds)
            max_retries: Maximum API call retries
            prompt_template: External prompt template file path
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.debug = debug
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_template = prompt_template
        self.llm_provider = None

        # Verify model parameter is provided
        if self.model is None:
            raise ValueError(
                f"Model name must be specified, please use --model parameter"
            )

        # Initialize LLM provider
        self._init_llm_provider()

    def _init_llm_provider(self):
        """Initialize LLM provider"""
        try:
            self.llm_provider = create_llm_provider(
                provider_name=self.provider,
                model=self.model,
                api_key=self.api_key,
                debug=self.debug,
            )
            print(
                f"Successfully initialized {self.provider} provider, model: {self.model}"
            )
        except Exception as e:
            print(f"Failed to initialize LLM provider: {e}")
            raise

    def truncate_oszicar(
        self, filepath: str, n_steps: int = 50
    ) -> Tuple[str, int, bool]:
        """
        Truncate OSZICAR file to first n_steps steps

        Args:
            filepath: OSZICAR file path
            n_steps: Number of steps to truncate

        Returns:
            (truncated content, actual steps, whether reached truncation steps)
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                return "", 0, False

            # Keep header
            header = lines[0]
            content_lines = [header]

            # Extract data lines (lines starting with RMM:, CG:, DAV:)
            data_lines = []
            for line in lines[1:]:
                line_stripped = line.strip()
                if line_stripped.startswith(("RMM:", "CG:", "DAV:")):
                    data_lines.append(line)
                    if len(data_lines) >= n_steps:
                        break

            content_lines.extend(data_lines)

            return "".join(content_lines), len(data_lines), len(data_lines) >= n_steps

        except Exception as e:
            print(f"Failed to read file {filepath}: {e}")
            return "", 0, False

    def load_prompt_template(self, template_path: str) -> Optional[str]:
        """
        Load external prompt template file

        Args:
            template_path: Template file path

        Returns:
            Template content string
        """
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Failed to load prompt template: {e}")
            return None

    def create_prompt(
        self, oszicar_content: str, n_steps: int, filepath: str = None
    ) -> str:
        """
        Create prompt to send to LLM

        Args:
            oszicar_content: OSZICAR file content
            n_steps: Number of truncated steps
            filepath: OSZICAR file path (optional)

        Returns:
            Complete prompt string
        """
        # Prioritize external template
        if self.prompt_template:
            template_content = self.load_prompt_template(self.prompt_template)
            if template_content:
                # Replace template variables
                prompt = template_content
                prompt = prompt.replace("{oszicar_content}", oszicar_content)
                prompt = prompt.replace("{n_steps}", str(n_steps))
                if filepath:
                    prompt = prompt.replace("{filepath}", filepath)
                else:
                    prompt = prompt.replace("{filepath}", "unknown")
                return prompt
            else:
                print(
                    f"Warning: Unable to load template {self.prompt_template}, using default prompt"
                )

        # Default prompt
        prompt = f"""You are an expert in computational materials science and VASP calculations. Your task is to analyze VASP OSZICAR files from single-point energy calculations to predict whether a calculation will ultimately converge.

OSZICAR file format:
- First line is header: N(electronic steps) E0(total energy) dE(energy change) d eps ncg rms rms(c)
- Subsequent lines represent electronic step results

Key indicators for convergence:
- Convergence: Both energy change (dE) and charge density residual (rms) show consistent downward trends
- Divergence: Either dE or rms fails to decrease or shows persistent oscillations

Here is the first {n_steps} steps of OSZICAR data from a single-point energy calculation:
{oszicar_content}

Based on this data, predict whether this calculation will ultimately converge.

Your response MUST be in the following single-line format:
PREDICTION=number CONFIDENCE=number REASONING=brief_reason

Where:
- PREDICTION: 1 for 'converges', 0 for 'diverges'
- CONFIDENCE: integer between 0 and 9 (9=extremely confident, 0=pure guess)
- REASONING: brief explanation for your prediction

Output only this single line, no additional text."""
        return prompt

    def call_llm_api(
        self,
        prompt: str,
        temperature: float = 0.1,
        timeout: int = 20,
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        Call LLM API

        Args:
            prompt: Prompt to send to LLM
            temperature: Temperature parameter
            timeout: Timeout (seconds)
            max_retries: Maximum retries

        Returns:
            LLM response text, returns None on failure
        """
        return self.llm_provider.call_api(prompt, temperature, timeout, max_retries)

    def parse_llm_response(
        self, response: str
    ) -> Tuple[Optional[int], Optional[int], str]:
        """
        Parse LLM response, extract prediction result and confidence

        Args:
            response: LLM response text

        Returns:
            (prediction result, confidence, reasoning)
            prediction result: 1 indicates convergence, 0 indicates non-convergence, None indicates parse failure
            confidence: integer 0-9, None indicates parse failure
        """
        if self.debug:
            print(f"\n=== DEBUG: Starting to parse response ===")
            print(f"Response content:\n{response}")

        if not response:
            if self.debug:
                print("Response is empty")
            return None, None, ""

        try:
            response = response.strip()

            # Support flexible matching of single-line format
            patterns = [
                r"PREDICTION=(\d)\s+CONFIDENCE=(\d)\s+REASONING=(.+)",
                r"PREDICTION\s*=\s*(\d)\s+CONFIDENCE\s*=\s*(\d)\s+REASONING\s*=\s*(.+)",
                r"prediction\s*=\s*(\d).*confidence\s*=\s*(\d).*reasoning\s*=\s*(.+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    prediction = int(match.group(1))
                    confidence = int(match.group(2))
                    reasoning = match.group(3).strip()

                    if prediction in [0, 1] and 0 <= confidence <= 9:
                        if self.debug:
                            print(
                                f"Parse successful: PREDICTION={prediction}, CONFIDENCE={confidence}, REASONING={reasoning[:100]}..."
                            )
                        return prediction, confidence, reasoning

            # If direct match fails, try to extract separately
            prediction = None
            confidence = None
            reasoning = ""

            # Extract prediction value
            pred_match = re.search(r"PREDICTION\s*=\s*(\d)", response, re.IGNORECASE)
            if pred_match:
                prediction = int(pred_match.group(1))
                if prediction not in [0, 1]:
                    prediction = None

            # Extract confidence
            conf_match = re.search(r"CONFIDENCE\s*=\s*(\d)", response, re.IGNORECASE)
            if conf_match:
                confidence = int(conf_match.group(1))
                if not (0 <= confidence <= 9):
                    confidence = None

            # Extract reasoning
            reason_match = re.search(
                r"REASONING\s*=\s*(.+)", response, re.IGNORECASE | re.DOTALL
            )
            if reason_match:
                reasoning = reason_match.group(1).strip()

            # If partially successful, return available information
            if prediction is not None and confidence is not None:
                return prediction, confidence, reasoning
            elif prediction is not None:
                return prediction, 5, reasoning

        except Exception as e:
            if self.debug:
                print(f"Parse failed: {e}")

        # Text matching as last fallback
        response_lower = response.lower()
        if "converge" in response_lower and "diverge" not in response_lower:
            if self.debug:
                print("Text matching result: Convergence (1)")
            return 1, 5, response
        elif "diverge" in response_lower or "not converge" in response_lower:
            if self.debug:
                print("Text matching result: Non-convergence (0)")
            return 0, 5, response

        if self.debug:
            print("Parse failed: Unable to recognize prediction result")
        return None, None, response

    def evaluate_single_sample(
        self, filepath: str, true_label: int, n_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate single sample

        Args:
            filepath: OSZICAR file path
            true_label: True label (0 or 1)
            n_steps: Number of truncated steps

        Returns:
            Evaluation result dictionary
        """
        result = {
            "filepath": filepath,
            "true_label": true_label,
            "n_steps": n_steps,
            "prediction": None,
            "reasoning": "",
            "roc_probability": None,
            "success": False,
            "error": "",
            "api_time": 0,
        }

        # Truncate OSZICAR data
        oszicar_content, actual_steps, enough_steps = self.truncate_oszicar(
            filepath, n_steps
        )
        if not oszicar_content:
            result["error"] = "Unable to read OSZICAR file"
            return result

        # Create prompt
        prompt = self.create_prompt(oszicar_content, n_steps, filepath)

        # Call LLM API
        start_time = time.time()
        response = self.call_llm_api(
            prompt, timeout=self.timeout, max_retries=self.max_retries
        )
        api_time = time.time() - start_time

        if not response:
            result["error"] = "API call failed"
            result["api_time"] = api_time
            return result

        # Parse response
        prediction, confidence_level, reasoning = self.parse_llm_response(response)

        # Fill results
        result["prediction"] = prediction
        result["confidence_score"] = confidence_level
        result["reasoning"] = reasoning
        result["api_time"] = api_time
        result["actual_steps"] = actual_steps
        result["enough_steps"] = enough_steps

        # Only save original confidence score, ROC probability calculated in post-processing stage
        if prediction is not None and confidence_level is not None:
            result["success"] = True
        else:
            result["error"] = "Unable to parse LLM response"
            result["raw_response"] = response  # Save raw response for debugging

        return result

    def generate_single_prompt(self, oszicar_path: str, n_steps: int = 50):
        """
        Generate prompt for single OSZICAR file

        Args:
            oszicar_path: OSZICAR file path
            n_steps: Number of truncated steps
        """
        if not os.path.exists(oszicar_path):
            print(f"Error: File does not exist: {oszicar_path}")
            return

        print(f"Generating prompt for file: {oszicar_path}")
        print(f"Truncated steps: {n_steps}")
        print("-" * 50)

        # Read OSZICAR content
        try:
            with open(oszicar_path, "r") as f:
                oszicar_content = f.read()

            # Calculate actual steps (minus header line)
            lines = oszicar_content.strip().split("\n")
            actual_steps = len(lines) - 1  # Minus header line

            # Truncate first n_steps steps
            if len(lines) > n_steps + 1:  # +1 for header
                oszicar_content = "\n".join(lines[: n_steps + 1])

            # Generate prompt
            prompt = self.create_prompt(oszicar_content, n_steps, oszicar_path)

            # Output information
            print(f"Actual steps: {actual_steps} (requested: {n_steps} steps)")
            print("-" * 30)
            print("Prompt content:")
            print(prompt)
            print("=" * 50)

        except Exception as e:
            print(f"Failed to read file: {e}")

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset

        Args:
            dataset_path: Dataset file path

        Returns:
            Dataset DataFrame
        """
        try:
            df = pd.read_csv(dataset_path)
            print(f"Successfully loaded dataset: {len(df)} samples")
            return df
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return pd.DataFrame()

    def filter_samples_for_n_steps(
        self, df: pd.DataFrame, n_steps: int
    ) -> pd.DataFrame:
        """
        Filter suitable samples based on n_steps (using step information in dataset)

        Args:
            df: Dataset DataFrame
            n_steps: Test steps

        Returns:
            Filtered DataFrame
        """
        # Check if dataset contains step information
        if "num_steps" not in df.columns:
            print(
                "Warning: Dataset does not contain step information, using file parsing method"
            )
            return self._filter_samples_by_file_parsing(df, n_steps)

        # Filter using step information in dataset
        filtered_df = df[df["num_steps"] >= n_steps].copy()

        # Verify files exist
        existing_files = []
        for _, row in filtered_df.iterrows():
            if os.path.exists(row["filepath"]):
                existing_files.append(row)
            elif self.debug:
                print(
                    f"Skipping sample {os.path.basename(row['filepath'])}: File does not exist"
                )

        final_df = pd.DataFrame(existing_files)
        print(
            f"Filtered {len(final_df)} samples suitable for {n_steps}-step testing from {len(df)} samples"
        )
        return final_df

    def _filter_samples_by_file_parsing(
        self, df: pd.DataFrame, n_steps: int
    ) -> pd.DataFrame:
        """
        Filter samples by parsing files (backup method)

        Args:
            df: Dataset DataFrame
            n_steps: Test steps

        Returns:
            Filtered DataFrame
        """
        filtered_samples = []

        for _, row in df.iterrows():
            filepath = row["filepath"]
            if os.path.exists(filepath):
                # Parse actual steps
                num_steps = self._parse_oszicar_steps_from_file(filepath)
                if num_steps is not None and num_steps >= n_steps:
                    filtered_samples.append(row)
                elif self.debug:
                    print(
                        f"Skipping sample {os.path.basename(filepath)}: Actual steps {num_steps} < test steps {n_steps}"
                    )
            elif self.debug:
                print(
                    f"Skipping sample {os.path.basename(filepath)}: File does not exist"
                )

        filtered_df = pd.DataFrame(filtered_samples)
        print(
            f"Filtered {len(filtered_df)} samples suitable for {n_steps}-step testing from {len(df)} samples"
        )
        return filtered_df

    def _parse_oszicar_steps_from_file(self, filepath: str) -> Optional[int]:
        """
        Parse actual steps from OSZICAR file (backup method)

        Args:
            filepath: OSZICAR file path

        Returns:
            Actual steps, returns None on parse failure
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                return None

            # Find the last line containing steps
            for line in reversed(lines[1:]):  # Skip header
                line_stripped = line.strip()
                if line_stripped.startswith(("RMM:", "CG:", "DAV:")):
                    # Extract steps
                    match = re.match(r"^(?:RMM|CG|DAV):\s*(\d+)", line_stripped)
                    if match:
                        return int(match.group(1))

            return None

        except Exception as e:
            if self.debug:
                print(f"Failed to parse steps from {filepath}: {e}")
            return None

    def run_evaluation(
        self,
        dataset_path: str,
        n_steps: int = 50,
        max_samples: int = None,
        output_file: str = None,
        call_interval: float = 0.5,
        console_log_file: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Run complete evaluation

        Args:
            dataset_path: Dataset file path
            n_steps: Number of truncated steps
            max_samples: Maximum test sample count (None means all)
            output_file: Result output file path
            call_interval: API call interval time (seconds)
            console_log_file: Console output log file path (optional)

        Returns:
            Evaluation results list
        """
        # Initialize console log
        console_log_lines = []

        # Load dataset
        df = self.load_dataset(dataset_path)
        if df.empty:
            return []

        # Filter samples based on n_steps
        df = self.filter_samples_for_n_steps(df, n_steps)
        if df.empty:
            msg = f"No samples found suitable for {n_steps}-step testing"
            print(msg)
            console_log_lines.append(msg)
            return []

        # Limit sample count
        if max_samples and max_samples < len(df):
            df = df.iloc[:max_samples]
            msg = f"Selected {max_samples} samples for testing"
            print(msg)
            console_log_lines.append(msg)

        results = []
        total_samples = len(df)

        header_line = (
            f"Starting evaluation of {total_samples} samples, steps: {n_steps}"
        )
        table_header = f"{'#':5} {'Status':4} {'Sample':40} {'Prediction':12} {'Truth':12} {'Steps':5} {'Confidence':10} {'Time(s)':6}"
        separator = "-" * 93

        print(header_line)
        print(table_header)
        print(separator)

        console_log_lines.extend([header_line, table_header, separator])

        for idx, (_, row) in enumerate(df.iterrows()):
            filepath = row["filepath"]
            true_label = row["label"]
            actual_steps = row.get("num_steps", "Unknown")

            result = self.evaluate_single_sample(filepath, true_label, n_steps)
            results.append(result)

            # Last part of filename
            filename = os.path.basename(filepath)
            dirname = os.path.basename(os.path.dirname(filepath))
            short_name = f"{dirname}/{filename}"

            if result["success"]:
                pred_label = "Converge" if result["prediction"] == 1 else "Diverge"
                true_label_str = "Converge" if true_label == 1 else "Diverge"
                confidence_score = result.get("confidence_score", "N/A")

                # Determine if prediction is correct
                is_correct = result["prediction"] == true_label
                color_code = "\033[92m" if is_correct else "\033[91m"  # Green/red
                reset_code = "\033[0m"
                status = "PASS" if is_correct else "FAIL"

                console_output = f"{idx + 1:5d} {color_code}{status:4}{reset_code} {short_name:40} {pred_label:12} {true_label_str:12} {actual_steps:5} {confidence_score:10} {result['api_time']:6.2f}"
                log_output = f"{idx + 1:5d} {status:4} {short_name:40} {pred_label:12} {true_label_str:12} {actual_steps:5} {confidence_score:10} {result['api_time']:6.2f}"

                print(console_output)
                console_log_lines.append(log_output)
            else:
                console_output = f"{idx + 1:5d} FAIL {short_name:40} -            -            -     -        -"
                print(console_output)
                console_log_lines.append(console_output)

            # Avoid API calls being too frequent
            if idx < total_samples - 1:
                time.sleep(call_interval)

        # Save results
        if output_file:
            self.save_results(results, output_file)

        # Save console log
        if console_log_file:
            self.save_console_log(console_log_lines, console_log_file)

        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save evaluation results

        Args:
            results: Evaluation results list
            output_file: Output file path
        """
        try:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    def save_console_log(self, log_lines: List[str], output_file: str):
        """
        Save console output log

        Args:
            log_lines: Log lines list
            output_file: Output file path
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for line in log_lines:
                    f.write(line + "\n")
            print(f"Console log saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save console log: {e}")

    def save_metrics_to_csv(self, metrics: Dict[str, float], output_file: str):
        """
        Save performance metrics to CSV file

        Args:
            metrics: Performance metrics dictionary
            output_file: Output file path
        """
        try:
            # Create single-row DataFrame
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"Performance metrics saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save performance metrics: {e}")

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics

        Args:
            results: Evaluation results list

        Returns:
            Performance metrics dictionary
        """
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            return {}

        true_labels = [r["true_label"] for r in successful_results]
        predictions = [r["prediction"] for r in successful_results]

        # Calculate basic metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        metrics = {
            "total_samples": len(results),
            "successful_samples": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "accuracy": accuracy_score(true_labels, predictions),
            "precision": precision_score(true_labels, predictions, zero_division=0),
            "recall": recall_score(true_labels, predictions, zero_division=0),
            "f1_score": f1_score(true_labels, predictions, zero_division=0),
        }

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(true_labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

        # Calculate average API call time
        api_times = [r["api_time"] for r in successful_results]
        if api_times:
            metrics["avg_api_time"] = sum(api_times) / len(api_times)
            metrics["min_api_time"] = min(api_times)
            metrics["max_api_time"] = max(api_times)

        # Correct AUC calculation - use unified ROC calculation tool
        if successful_results:
            try:
                from sklearn.metrics import roc_auc_score
                from roc_utils import get_roc_probabilities

                # Convert to DataFrame to use unified tool
                results_df = pd.DataFrame(successful_results)

                # Use unified ROC probability calculation
                roc_probs = get_roc_probabilities(results_df)
                auc = roc_auc_score(true_labels, roc_probs)
                metrics["auc"] = auc

            except Exception as e:
                if self.debug:
                    print(f"AUC calculation failed: {e}")
                # Cannot calculate AUC with single-class samples
                pass

        return metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LLM VASP Convergence Prediction Evaluation"
    )
    parser.add_argument(
        "--dataset", "-d", default="benchmark_dataset.csv", help="Dataset file path"
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=50,
        help="Number of truncated steps (50, 100, 150)",
    )
    parser.add_argument(
        "--max-samples", "-m", type=int, default=None, help="Maximum test sample count"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=f"llm_results_{int(time.time())}.csv",
        help="Result output file path",
    )
    parser.add_argument(
        "--metrics-output", help="Performance metrics output file path (optional)"
    )
    parser.add_argument("--console-log", help="Console output log file path (optional)")
    parser.add_argument(
        "--provider",
        default="zhipuai",
        choices=["zhipuai", "siliconflow", "openrouter", "litellm", "deepseek"],
        help="LLM provider (zhipuai, siliconflow, openrouter, litellm, deepseek)",
    )
    parser.add_argument(
        "--model", required=True, help="Model name to use (required parameter)"
    )
    parser.add_argument(
        "--api-key",
        help="API key (if not specified, uses default value or environment variable)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, display detailed output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API call timeout (seconds), default 60 seconds (1 minute)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API call retries, default 3 times",
    )
    parser.add_argument(
        "--call-interval",
        type=float,
        default=0.5,
        help="API call interval time (seconds), default 0.5 seconds",
    )
    parser.add_argument(
        "--generate-prompt",
        help="Generate prompt without calling API. Must specify OSZICAR file path",
    )
    parser.add_argument(
        "--prompt-template",
        help="External prompt template file path (recommended: prompt_templates/zero_shot.prompt.md, prompt_templates/few_shot_1.prompt.md, prompt_templates/few_shot_2.prompt.md, prompt_templates/few_shot_3.prompt.md)",
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = LLMConvergenceEvaluator(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        debug=args.debug,
        timeout=args.timeout,
        max_retries=args.max_retries,
        prompt_template=args.prompt_template,
    )

    # Generate prompt mode
    if args.generate_prompt:
        evaluator.generate_single_prompt(
            oszicar_path=args.generate_prompt, n_steps=args.steps
        )
        return

    # Run evaluation
    results = evaluator.run_evaluation(
        dataset_path=args.dataset,
        n_steps=args.steps,
        max_samples=args.max_samples,
        output_file=args.output,
        call_interval=args.call_interval,
        console_log_file=args.console_log,
    )

    # Calculate and display performance metrics
    if results:
        metrics = evaluator.calculate_metrics(results)
        print("\n" + "=" * 60)
        print("Evaluation Summary Report")
        print("=" * 60)
        if metrics:
            print(f"Total Samples:      {metrics['total_samples']}")
            print(
                f"Successful Samples: {metrics['successful_samples']} ({metrics['success_rate']:.1%})"
            )
            print(
                f"Accuracy:           {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})"
            )
            print(f"Precision:          {metrics['precision']:.3f}")
            print(f"Recall:             {metrics['recall']:.3f}")
            print(f"F1-Score:           {metrics['f1_score']:.3f}")
            if "auc" in metrics:
                print(f"AUC:                {metrics['auc']:.3f}")

            # Confusion matrix
            if "true_positives" in metrics:
                print(f"\nConfusion Matrix:")
                print(
                    f"  TP: {metrics['true_positives']:3d}  FP: {metrics['false_positives']:3d}"
                )
                print(
                    f"  FN: {metrics['false_negatives']:3d}  TN: {metrics['true_negatives']:3d}"
                )

            # API performance
            if "avg_api_time" in metrics:
                print(f"\nAPI Performance:")
                print(f"  Average: {metrics['avg_api_time']:.2f}s")
                print(f"  Min:     {metrics['min_api_time']:.2f}s")
                print(f"  Max:     {metrics['max_api_time']:.2f}s")

            # Save performance metrics to CSV file
            if args.metrics_output:
                # Add model information and timestamp
                metrics["model"] = args.model
                metrics["provider"] = args.provider
                metrics["n_steps"] = args.steps
                metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                evaluator.save_metrics_to_csv(metrics, args.metrics_output)

            print("-" * 93)
            print(
                f"{'#':5} {'Status':4} {'Sample':40} {'Prediction':12} {'Truth':12} {'Steps':5} {'Confidence':10} {'Time(s)':6}"
            )
            print(f"\n💡 Results analysis commands:")
            print(f"  python scripts/analyze_results.py {args.output}")
            print(
                f"  python scripts/analyze_results.py {args.output} --model {args.model}"
            )
            if args.metrics_output:
                print(f"  # Performance metrics saved to: {args.metrics_output}")
            print(f"")
            print(f"Analysis options:")
            print(f"  --compare     Compare multiple model performance")
            print(f"  --output-prefix Define custom output prefix")
        else:
            print("No successful samples")


if __name__ == "__main__":
    main()

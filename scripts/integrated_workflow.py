#!/usr/bin/env python3
"""
LLM VASP Convergence Dynamic Prediction Evaluation Script

This script extends the LLM VASP convergence prediction evaluation by introducing a dynamic assessment mechanism.
For OSZICAR files longer than the initial step count, if the LLM's prediction confidence is below the set threshold,
the script will incrementally increase the number of steps read and resubmit to the LLM until achieving high confidence prediction,
reaching the maximum step count, or insufficient sample steps for further increase.
"""

import os
import sys
import json
import re
import pandas as pd
import time
from typing import Tuple, Optional, List, Dict, Any
import argparse
import signal # Import signal module for capturing Ctrl+C
import functools # Import functools for partial

# Add src directory to Python path
# Assume current script is located at your_project_root/
# and llm_providers.py and evaluate_llm_convergence.py are in your_project_root/src/
# or evaluate_llm_convergence.py is also in your_project_root/
# Adjust path to ensure correctness
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..')) # Assume src is one level up
if not os.path.exists(os.path.join(project_root, 'src')): # If src is in current directory
    project_root = current_script_dir

if os.path.exists(os.path.join(project_root, 'src')):
    sys.path.insert(0, os.path.join(project_root, 'src'))
else: # If src directory doesn't exist, assume llm_providers.py and evaluate_llm_convergence.py are in current directory or PATH
    print("Warning: 'src' directory not found. Please ensure 'llm_providers.py' and 'evaluate_llm_convergence.py' are in Python path.")

# Try to import, will error if not found
try:
    from llm_providers import create_llm_provider
    # Assume evaluate_llm_convergence.py is in the same directory or src directory
    # If it's in project root or src directory, can import by adjusting sys.path
    # If BaseLLMConvergenceEvaluator is actually in evaluate_llm_convergence.py, and evaluate_llm_convergence.py is in project root
    # then import method may need to be adjusted to from .evaluate_llm_convergence import LLMConvergenceEvaluator
    # But according to previous convention, it's in src directory
    from evaluate_llm_convergence import LLMConvergenceEvaluator as BaseLLMConvergenceEvaluator
except ImportError as e:
    print(f"Failed to import module: {e}")
    print("Please ensure 'llm_providers.py' and 'evaluate_llm_convergence.py' files exist in the correct path.")
    print(f"Current Python search path: {sys.path}")
    sys.exit(1)


class DynamicLLMConvergenceEvaluator(BaseLLMConvergenceEvaluator):
    """
    LLM Convergence Dynamic Prediction Evaluator
    Inherits from LLMConvergenceEvaluator, adds dynamic evaluation logic
    """

    def __init__(self, provider: str = "zhipuai", model: str = None, api_key: str = None,
                 debug: bool = False, timeout: int = 60, max_retries: int = 3,
                 prompt_template: str = None, confidence_cutoff: int = 7,
                 step_increments: List[int] = None, max_successful_evals: Optional[int] = None): # New max_successful_evals parameter
        """
        Initialize dynamic evaluator

        Args:
            provider: LLM provider name
            model: Model name to use
            api_key: API key
            debug: Whether to enable debug mode
            timeout: API call timeout (seconds)
            max_retries: Maximum API call retries
            prompt_template: External prompt template file path
            confidence_cutoff: Confidence threshold (0-9), will increase steps for re-evaluation if below this value
            step_increments: List of step increments per iteration. For example: [50, 70, 90, 120, 150, 170]
            max_successful_evals: Maximum number of samples with successful evaluation and definitive results (Continue/Kill/Insufficient_Steps/Unresolved_MaxSteps).
                                  Script will stop after reaching this count. None means no limit.
        """
        super().__init__(provider, model, api_key, debug, timeout, max_retries, prompt_template)
        self.confidence_cutoff = confidence_cutoff
        self.step_increments = step_increments if step_increments is not None else [50, 70, 90, 120, 150, 170]
        # Ensure step increments are sorted
        self.step_increments = sorted(list(set(self.step_increments)))
        self.current_run_results = [] # Store all results from current run
        self.max_successful_evals = max_successful_evals # New parameter
        self.output_file = None # Set in run_dynamic_evaluation, used for saving on Ctrl+C
        self.call_interval = 0.5 # Default value, updated in run_dynamic_evaluation

        print(f"Dynamic evaluator initialized: confidence threshold={self.confidence_cutoff}, step increments={self.step_increments}")
        if self.max_successful_evals is not None:
            print(f"Will stop after {self.max_successful_evals} successfully evaluated samples.")

    def dynamic_evaluate_single_sample(self, filepath: str, true_label: int,
                                       initial_n_steps: int = 50, actual_total_steps: int = None) -> Dict[str, Any]:
        """
        Dynamically evaluate single sample, incrementally increasing read steps based on LLM confidence.

        Args:
            filepath: OSZICAR file path
            true_label: True label (0 or 1)
            initial_n_steps: Initial evaluation step count
            actual_total_steps: Actual total steps of sample (from dataset, if exists)

        Returns:
            Final evaluation result dictionary
        """
        if actual_total_steps is None:
            # If not provided, try to parse from file
            actual_total_steps = self._parse_oszicar_steps_from_file(filepath)
            if actual_total_steps is None:
                print(f"  ❌ Error: Unable to parse actual total steps from file {filepath}.")
                return {
                    'filepath': filepath,
                    'true_label': true_label,
                    'total_steps': None,  # New field
                    'final_prediction': None,
                    'final_confidence': None,
                    'final_reasoning': "Unable to parse actual total steps",
                    'status': 'Error_ParsingTotalSteps', # More specific error status
                    'predicted_at_steps': None,
                    'final_api_time': 0,
                    'evaluation_path': [],
                    'num_api_calls': 0
                }

        sample_eval_path = [] # Record evaluation path for this sample
        num_api_calls = 0
        total_api_time = 0

        # Find appropriate starting step count from step increments list
        # At this point self.step_increments is already sorted, and initial_n_steps should be included
        current_step_idx = 0
        while current_step_idx < len(self.step_increments) and self.step_increments[current_step_idx] < initial_n_steps:
            current_step_idx += 1

        # Ensure current_read_steps doesn't go out of bounds
        if current_step_idx >= len(self.step_increments):
            # This theoretically shouldn't happen, as main function ensures initial_n_steps is included and step_increments is sorted
            # If it does happen, it means initial_n_steps is larger than all increments, and increment list only contains smaller numbers
            print(f"  ⚠️ Warning: initial_n_steps ({initial_n_steps}) is larger than all configured step increments. Will use maximum increment {self.step_increments[-1]}.")
            current_read_steps = self.step_increments[-1]
            current_step_idx = len(self.step_increments) - 1
        else:
            current_read_steps = self.step_increments[current_step_idx]


        while True:
            # Check if current read steps exceed sample total steps
            if current_read_steps > actual_total_steps:
                status = 'Insufficient_Steps'
                final_result = {
                    'filepath': filepath,
                    'true_label': true_label,
                    'total_steps': actual_total_steps,  # New field
                    'final_prediction': None,
                    'final_confidence': None,
                    'final_reasoning': f"When evaluating at {current_read_steps} steps, actual steps only {actual_total_steps}, insufficient data.",
                    'status': status,
                    'predicted_at_steps': actual_total_steps, # Record actual maximum readable steps
                    'final_api_time': total_api_time,
                    'evaluation_path': sample_eval_path,
                    'num_api_calls': num_api_calls
                }
                sample_eval_path.append({'steps': current_read_steps, 'prediction': None, 'confidence': None, 'status': status})
                print(f"    ⚠️ Actual steps {actual_total_steps} insufficient to evaluate {current_read_steps} steps, marked as {status}.")
                return final_result

            print(f"  -> Attempting {filepath} (truth: {true_label}) - reading first {current_read_steps}/{actual_total_steps} steps...")

            single_eval_result = self.evaluate_single_sample(filepath, true_label, current_read_steps)
            num_api_calls += 1
            total_api_time += single_eval_result['api_time']

            eval_entry = {
                'steps': current_read_steps,
                'prediction': single_eval_result['prediction'],
                'confidence': single_eval_result.get('confidence_score'),
                'status': 'API_Success' if single_eval_result['success'] else 'API_Failed'
            }
            sample_eval_path.append(eval_entry)

            if not single_eval_result['success']:
                print(f"    ❌ API call failed or parsing error, skipping current step count. Error: {single_eval_result['error']}")
                # API call failed, try next step increment.
                # Note: If multiple API failures prevent progress, will eventually enter Unresolved_MaxSteps
                prediction_at_fail = None # No valid prediction on failure
                confidence_at_fail = None
                reasoning_at_fail = single_eval_result['error']
            else:
                prediction_at_fail = single_eval_result['prediction']
                confidence_at_fail = single_eval_result['confidence_score']
                reasoning_at_fail = single_eval_result['reasoning']

                print(f"    LLM prediction: PREDICTION={prediction_at_fail}, CONFIDENCE={confidence_at_fail}, REASONING='{reasoning_at_fail[:50]}...'")

                if confidence_at_fail is not None and confidence_at_fail >= self.confidence_cutoff:
                    # Reached confidence requirement, end evaluation
                    status = 'Continue' if prediction_at_fail == 1 else 'Kill'
                    print(f"    ✅ Reached confidence threshold ({confidence_at_fail} >= {self.confidence_cutoff}), marked as {status}")
                    final_result = {
                        'filepath': filepath,
                        'true_label': true_label,
                        'total_steps': actual_total_steps,  # New field
                        'final_prediction': prediction_at_fail,
                        'final_confidence': confidence_at_fail,
                        'final_reasoning': reasoning_at_fail,
                        'status': status,
                        'predicted_at_steps': current_read_steps,
                        'final_api_time': total_api_time,
                        'evaluation_path': sample_eval_path,
                        'num_api_calls': num_api_calls
                    }
                    return final_result
                else:
                    print(f"    ⚠️ Insufficient confidence ({confidence_at_fail} < {self.confidence_cutoff}), increasing steps for continued evaluation.")

            # Increase steps
            current_step_idx += 1
            if current_step_idx < len(self.step_increments):
                current_read_steps = self.step_increments[current_step_idx]
            else:
                # No more step increments, and failed to reach confidence or API failed
                status = 'Unresolved_MaxSteps'
                print(f"    🚫 Reached maximum evaluation steps ({self.step_increments[-1]} steps), failed to reach confidence. Marked as {status}")
                final_result = {
                    'filepath': filepath,
                    'true_label': true_label,
                    'total_steps': actual_total_steps,  # New field
                    'final_prediction': prediction_at_fail, # Use last prediction
                    'final_confidence': confidence_at_fail, # Use last confidence
                    'final_reasoning': reasoning_at_fail,
                    'status': status,
                    'predicted_at_steps': current_read_steps,
                    'final_api_time': total_api_time,
                    'evaluation_path': sample_eval_path,
                    'num_api_calls': num_api_calls
                }
                return final_result

            # API call interval
            time.sleep(self.call_interval)

    def run_dynamic_evaluation(self, dataset_path: str, initial_n_steps: int = 50,
                               max_samples: Optional[int] = None, output_file: str = None,
                               call_interval: float = 0.5, console_log_file: str = None,
                               restart_from_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run complete dynamic evaluation process

        Args:
            dataset_path: Dataset file path
            initial_n_steps: Initial evaluation step count
            max_samples: Maximum test sample count (None means all)
            output_file: Result output file path
            call_interval: API call interval time (seconds)
            console_log_file: Console output log file path (optional)
            restart_from_file: Continue from previously saved results file

        Returns:
            Evaluation results list
        """
        self.call_interval = call_interval # Store interval time for internal use
        self.output_file = output_file # Store output file path for saving on Ctrl+C

        # Initialize console log
        console_log_lines = []

        # Load dataset
        df = self.load_dataset(dataset_path)
        if df.empty:
            return []

        # Filter samples with steps greater than initial steps
        df_filtered = df[df['num_steps'] >= initial_n_steps].copy()

        # Verify files exist
        existing_files = []
        for _, row in df_filtered.iterrows():
            if os.path.exists(row['filepath']):
                existing_files.append(row)
            elif self.debug:
                print(f"Skipping sample {os.path.basename(row['filepath'])}: File does not exist")

        df_final = pd.DataFrame(existing_files)
        if df_final.empty:
            msg = f"No samples found with total steps >= {initial_n_steps} and existing files for dynamic testing."
            print(msg)
            console_log_lines.append(msg)
            return []

        # Handle restart mechanism
        processed_filepaths = set()
        if restart_from_file and os.path.exists(restart_from_file):
            try:
                restart_df = pd.read_csv(restart_from_file)
                # Define columns we want to read and retain from CSV, these are most columns in final output
                # 'evaluation_path' is usually not directly stored in CSV as it's a list structure
                expected_simple_columns = [
                    'filepath', 'true_label', 'total_steps',  # New total_steps
                    'final_prediction', 'final_confidence',
                    'final_reasoning', 'status', 'predicted_at_steps',
                    'final_api_time', 'num_api_calls'
                ]

                # Check if all expected columns are in restart file
                if all(col in restart_df.columns for col in expected_simple_columns):
                    for _, row in restart_df.iterrows():
                        result_dict = {col: row[col] for col in expected_simple_columns if col in row}
                        # Add empty evaluation_path list for compatibility with subsequent processing, though its content won't be used
                        result_dict['evaluation_path'] = []
                        self.current_run_results.append(result_dict)
                        processed_filepaths.add(row['filepath'])
                    print(f"Recovered from {restart_from_file}, already processed {len(processed_filepaths)} samples.")
                else:
                    print(f"Warning: Recovery file {restart_from_file} columns don't match expected format, will start from beginning.")
            except Exception as e:
                print(f"Warning: Unable to recover from {restart_from_file}: {e}. Will start from beginning.")

        # Filter out already processed samples
        samples_to_process = [row for _, row in df_final.iterrows() if row['filepath'] not in processed_filepaths]
        df_to_process = pd.DataFrame(samples_to_process)

        # Limit sample count (only count unprocessed samples)
        if max_samples and max_samples < len(df_to_process):
            df_to_process = df_to_process.iloc[:max_samples]
            msg = f"Selected {max_samples} unprocessed samples for testing"
            print(msg)
            console_log_lines.append(msg)

        total_samples_to_process = len(df_to_process)

        header_line = f"Starting dynamic evaluation of {total_samples_to_process} samples (out of {len(df_final)}), initial steps: {initial_n_steps}"
        # Update table header, add Total column
        table_header = f"{'#':5} {'Status':20} {'Sample':40} {'Pred':5} {'Conf':5} {'AtSteps':7} {'Total':7} {'Truth':5} {'API_Calls':9} {'Time(s)':8}"
        separator = "-" * 118  # Increase separator length to match new header

        print(header_line)
        print(table_header)
        print(separator)

        console_log_lines.extend([header_line, table_header, separator])

        # Iterate through samples
        for idx, (_, row) in enumerate(df_to_process.iterrows()):
            # Check if maximum successful evaluation sample count reached
            # Successful evaluation defined as status not being Error_ParsingTotalSteps or API_Failed
            successful_eval_count = len([r for r in self.current_run_results if r.get('status') not in ['Error_ParsingTotalSteps', 'API_Failed', None]])
            if self.max_successful_evals is not None and successful_eval_count >= self.max_successful_evals:
                print(f"\nReached maximum successful evaluation sample count {self.max_successful_evals}, stopping evaluation.")
                break # Exit loop

            filepath = row['filepath']
            true_label = row['label']
            actual_total_steps = row.get('num_steps', None) # Ensure total steps are obtained

            print(f"\n--- Starting evaluation of sample {idx+1}/{total_samples_to_process} (successfully evaluated: {successful_eval_count}/{self.max_successful_evals if self.max_successful_evals else 'Unlimited'}): {os.path.basename(filepath)} ---")

            result = self.dynamic_evaluate_single_sample(filepath, true_label, initial_n_steps, actual_total_steps)
            self.current_run_results.append(result)

            # Last part of filename
            filename = os.path.basename(filepath)
            dirname = os.path.basename(os.path.dirname(filepath))
            short_name = f"{dirname}/{filename}"

            pred_label = "C" if result['final_prediction'] == 1 else "D" if result['final_prediction'] == 0 else "-"
            true_label_str = "C" if true_label == 1 else "D"
            confidence_score = result.get('final_confidence', '-')

            # Determine if prediction is correct
            is_correct = (result['final_prediction'] is not None and result['final_prediction'] == true_label)
            color_code = "\033[92m" if is_correct and result['status'] in ['Continue', 'Kill'] else "\033[91m" if not is_correct and result['status'] in ['Continue', 'Kill'] else "\033[0m" # Green/red/default
            reset_code = "\033[0m"

            # Update console output, add total_steps
            console_output = (
                f"{idx+1:5d} {color_code}{result['status']:20}{reset_code} {short_name:40} "
                f"{pred_label:5} {str(confidence_score):5} {str(result['predicted_at_steps']):7} "
                f"{str(result['total_steps']):7} {true_label_str:5} {result['num_api_calls']:9} {result['final_api_time']:8.2f}"
            )
            log_output = (
                f"{idx+1:5d} {result['status']:20} {short_name:40} "
                f"{pred_label:5} {str(confidence_score):5} {str(result['predicted_at_steps']):7} "
                f"{str(result['total_steps']):7} {true_label_str:5} {result['num_api_calls']:9} {result['final_api_time']:8.2f}"
            )

            print(console_output)
            console_log_lines.append(log_output)

            # Save results after processing each sample, especially useful for Ctrl+C and max_successful_evals
            if self.output_file:
                # Filter out evaluation_path as it's not easy to handle in CSV
                results_for_csv = [{k: v for k, v in res.items() if k != 'evaluation_path'} for res in self.current_run_results]
                self.save_results(results_for_csv, self.output_file)

            # API call interval
            # Only wait if not the last sample and haven't reached maximum evaluation count
            if idx < total_samples_to_process - 1 and \
               (self.max_successful_evals is None or successful_eval_count + 1 < self.max_successful_evals):
                time.sleep(call_interval)

        # At end of loop, save console log
        if console_log_file:
            self.save_console_log(console_log_lines, console_log_file)

        return self.current_run_results
    
    def calculate_dynamic_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate dynamic evaluation performance metrics

        Args:
            results: Dynamic evaluation results list

        Returns:
            Performance metrics dictionary
        """
        # Filter samples with successful predictions
        # Only consider samples with status 'Continue' or 'Kill' for accuracy and other metric calculations
        predicted_results = [r for r in results if r.get('status') in ['Continue', 'Kill']]

        if not predicted_results:
            return {
                'total_samples_processed': len(results),
                'total_predicted_samples': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'avg_api_calls_per_prediction': 0.0,
                'avg_steps_at_prediction': 0.0,
                'avg_api_time_per_prediction': 0.0,
                'status_distribution': {status: 0 for status in ['Continue', 'Kill', 'Insufficient_Steps', 'Unresolved_MaxSteps', 'Error_ParsingTotalSteps', 'API_Failed', 'Unknown']} # Add all possible statuses
            }

        true_labels = [r['true_label'] for r in predicted_results]
        predictions = [r['final_prediction'] for r in predicted_results]

        # Calculate basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'total_samples_processed': len(results),
            'total_predicted_samples': len(predicted_results),
            'prediction_rate': len(predicted_results) / len(results) if len(results) > 0 else 0,
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1_score': f1_score(true_labels, predictions, zero_division=0)
        }

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
        else: # Handle case with only one class
            metrics['true_negatives'] = 0
            metrics['false_positives'] = 0
            metrics['false_negatives'] = 0
            metrics['true_positives'] = len(true_labels) if (1 in true_labels and 1 in predictions) else 0 # Simple estimation

        # Dynamic system specific metrics
        num_api_calls_list = [r['num_api_calls'] for r in predicted_results]
        steps_at_prediction_list = [r['predicted_at_steps'] for r in predicted_results]
        api_time_list = [r['final_api_time'] for r in predicted_results]

        if num_api_calls_list:
            metrics['avg_api_calls_per_prediction'] = sum(num_api_calls_list) / len(num_api_calls_list)
            metrics['avg_steps_at_prediction'] = sum(steps_at_prediction_list) / len(steps_at_prediction_list)
            metrics['avg_api_time_per_prediction'] = sum(api_time_list) / len(api_time_list)
            metrics['max_api_calls'] = max(num_api_calls_list)
            metrics['min_api_calls'] = min(num_api_calls_list)
            metrics['max_steps_at_prediction'] = max(steps_at_prediction_list)
            metrics['min_steps_at_prediction'] = min(steps_at_prediction_list)
        else:
            metrics['avg_api_calls_per_prediction'] = 0.0
            metrics['avg_steps_at_prediction'] = 0.0
            metrics['avg_api_time_per_prediction'] = 0.0
            metrics['max_api_calls'] = 0
            metrics['min_api_calls'] = 0
            metrics['max_steps_at_prediction'] = 0
            metrics['min_steps_at_prediction'] = 0

        # Status distribution
        status_counts = {}
        for r in results:
            status = r.get('status', 'Unknown') # Error handling
            status_counts[status] = status_counts.get(status, 0) + 1
        metrics['status_distribution'] = status_counts

        return metrics

# Ctrl+C signal handler function
def signal_handler(sig, frame, evaluator_ref: DynamicLLMConvergenceEvaluator):
    print("\nCtrl+C detected. Attempting to save current results before exiting...")
    if evaluator_ref.output_file and evaluator_ref.current_run_results:
        # Filter out evaluation_path as it's not easy to handle in CSV
        results_for_csv = [{k: v for k, v in res.items() if k != 'evaluation_path'} for res in evaluator_ref.current_run_results]
        evaluator_ref.save_results(results_for_csv, evaluator_ref.output_file)
        print(f"Current results saved to: {evaluator_ref.output_file}")
    else:
        print("No output file specified or no results to save.")
    sys.exit(0)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LLM VASP Convergence Dynamic Prediction Evaluation')
    parser.add_argument('--dataset', '-d', default='benchmark_dataset.csv',
                       help='Dataset file path')
    parser.add_argument('--initial-steps', '-i', type=int, default=50,
                       help='Initial evaluation step count, filter samples with total steps greater than this value, and start dynamic evaluation from this step count')
    parser.add_argument('--step-increments', '-s', type=str, default="50,70,90,120,150,170",
                       help='Comma-separated list of step increments, e.g., "50,70,90,120,150,170"')
    parser.add_argument('--confidence-cutoff', '-c', type=int, default=7,
                       help='LLM prediction confidence threshold (0-9), will increase steps for re-evaluation if below this value')
    parser.add_argument('--max-samples', '-m', type=int, default=None,
                       help='Maximum test sample count (None means all)')
    parser.add_argument('--max-successful-evals', type=int, default=None, # New parameter
                       help='Maximum number of samples with successful evaluation and definitive results (Continue/Kill/Insufficient_Steps/Unresolved_MaxSteps). '
                            'Script will stop after reaching this count. None means no limit.')
    parser.add_argument(
        '--output', '-o',
        default=f'dynamic_llm_results_{int(time.time())}.csv',
        help='Result output file path'
    )
    parser.add_argument(
        '--console-log', type=str, default=None,
        help='Console output log file save path (optional)'
    )
    parser.add_argument(
        '--restart-from', type=str, default=None,
        help='Recover from previously saved results CSV file and continue running'
    )
    parser.add_argument(
        '--provider', type=str, default="zhipuai",
        help='LLM provider, e.g., zhipuai, openai'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model name to use (depends on provider)'
    )
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='Model API Key'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode, output more logs'
    )
    parser.add_argument(
        '--timeout', type=int, default=60,
        help='API call timeout (seconds)'
    )
    parser.add_argument(
        '--max-retries', type=int, default=3,
        help='Maximum API call retries'
    )
    parser.add_argument(
        '--prompt-template', type=str, default=None,
        help='External prompt template file path (optional)'
    )
    parser.add_argument(
        '--call-interval', type=float, default=0.5,
        help='API call interval time (seconds)'
    )

    args = parser.parse_args()

    # Parse step-increments parameter
    step_increments = [int(x.strip()) for x in args.step_increments.split(",") if x.strip().isdigit()]

    evaluator = DynamicLLMConvergenceEvaluator(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        debug=args.debug,
        timeout=args.timeout,
        max_retries=args.max_retries,
        prompt_template=args.prompt_template,
        confidence_cutoff=args.confidence_cutoff,
        step_increments=step_increments,
        max_successful_evals=args.max_successful_evals,
    )

    # Capture Ctrl+C
    signal.signal(signal.SIGINT, functools.partial(signal_handler, evaluator_ref=evaluator))

    # Run dynamic evaluation
    results = evaluator.run_dynamic_evaluation(
        dataset_path=args.dataset,
        initial_n_steps=args.initial_steps,
        max_samples=args.max_samples,
        output_file=args.output,
        call_interval=args.call_interval,
        console_log_file=args.console_log,
        restart_from_file=args.restart_from
    )

    # Calculate and print metrics
    metrics = evaluator.calculate_dynamic_metrics(results)
    print("\n=== Dynamic Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
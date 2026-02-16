#!/usr/bin/env python3
"""
VASP Convergence Prediction - Static Rule-Based Method
Predicts whether a calculation task will converge by analyzing the first N steps of OSZICAR files
using predefined physical rules. Used for comparison with LLM methods.
"""

import os
import sys
import json
import re
import pandas as pd
import time
from typing import Tuple, Optional, List, Dict, Any
import argparse
import numpy as np


class RuleBasedConvergenceEvaluator:
    """Rule-based convergence prediction evaluator"""

    def __init__(self, rule_config: str = None, debug: bool = False):
        """
        Initialize evaluator

        Args:
            rule_config: Rule configuration file path
            debug: Whether to enable debug mode
        """
        self.debug = debug
        self.rules = self.load_rules(rule_config)

        print(f"Initialized rule-based evaluator with {len(self.rules)} rules")
        if debug:
            for i, rule in enumerate(self.rules):
                print(f"  Rule {i+1}: {rule.get('name', f'Rule {i+1}')}")
    
    def load_rules(self, rule_config: str = None) -> List[Dict]:
        """
        Load rule configuration

        Args:
            rule_config: Rule configuration file path

        Returns:
            List of rules
        """
        default_rules = [
            {
                "name": "Energy convergence trend rule",
                "weight": 0.25,
                "check": "energy_trend"
            },
            {
                "name": "Charge density convergence trend rule",
                "weight": 0.30,
                "check": "charge_density_trend"
            },
            {
                "name": "Oscillation detection rule",
                "weight": 0.30,
                "check": "oscillation_check"
            },
            {
                "name": "Convergence threshold rule",
                "weight": 0.15,
                "check": "convergence_threshold"
            }
        ]

        if rule_config and os.path.exists(rule_config):
            try:
                with open(rule_config, 'r', encoding='utf-8') as f:
                    custom_rules = json.load(f)
                print(f"Loaded custom rules: {rule_config}")
                return custom_rules
            except Exception as e:
                print(f"Failed to load rule configuration file: {e}, using default rules")

        return default_rules
    
    def truncate_oszicar(self, filepath: str, n_steps: int = 50) -> Tuple[str, int, bool]:
        """
        Truncate OSZICAR file to first n_steps

        Args:
            filepath: OSZICAR file path
            n_steps: Number of steps to truncate

        Returns:
            (Truncated content, actual steps, whether reached truncation steps)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
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
                if line_stripped.startswith(('RMM:', 'CG:', 'DAV:')):
                    data_lines.append(line)
                    if len(data_lines) >= n_steps:
                        break

            content_lines.extend(data_lines)

            return ''.join(content_lines), len(data_lines), len(data_lines) >= n_steps

        except Exception as e:
            print(f"Failed to read file {filepath}: {e}")
            return "", 0, False
    
    def parse_oszicar_data(self, oszicar_content: str) -> Dict[str, List[float]]:
        """
        Parse OSZICAR data and extract key physical quantities

        Args:
            oszicar_content: OSZICAR file content

        Returns:
            Dictionary containing lists of physical quantities
        """
        data = {
            'step': [],
            'energy': [],
            'dE': [],
            'rms': [],
            'rms_c': []
        }
        
        lines = oszicar_content.strip().split('\n')
        if len(lines) <= 1:
            return data
        

        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Check if it's a data line (starting with RMM:, CG:, DAV:)
            if not line.startswith(('RMM:', 'CG:', 'DAV:')):
                continue

            try:
                # Use regex to match step number
                step_match = re.search(r'(RMM|CG|DAV):\s+(\d+)', line)
                if not step_match:
                    continue

                step = int(step_match.group(2))

                # Remove prefix and step number, keep numeric part
                # Example: "RMM:   1    -0.160814934095E+04   -0.16081E+04   -0.35957E+05  3888   0.165E+03"
                # Remove "RMM:   1    " part
                prefix_pattern = r'(RMM|CG|DAV):\s+\d+\s+'
                numbers_part = re.sub(prefix_pattern, '', line)

                # Extract all numbers
                numbers = []
                # Use finditer to find all scientific notation or floating point numbers
                for match in re.finditer(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', numbers_part):
                    try:
                        numbers.append(float(match.group()))
                    except ValueError:
                        continue

                # According to OSZICAR format, number order is: E, dE, d eps, ncg, rms, rms_c
                # ncg is an integer, but we also convert it to float
                if len(numbers) >= 5:  # At least need E, dE, d eps, ncg, rms
                    energy = numbers[0]  # E
                    dE_original = numbers[1]  # dE (original value, may be negative)
                    dE_abs = abs(dE_original)  # dE absolute value
                    rms = numbers[4]  # rms (skip d eps and ncg)

                    # rms_c may not exist in some lines
                    rms_c = numbers[5] if len(numbers) > 5 else 0.0

                    # Store data
                    data['step'].append(step)
                    data['energy'].append(energy)
                    data['dE'].append(dE_abs)
                    data['rms'].append(rms)
                    data['rms_c'].append(rms_c)

            except Exception as e:
                if self.debug:
                    print(f"Failed to parse line: {line}, error: {e}")
                continue
        
        return data
    
    def calculate_trend(self, values: List[float], window: int = 5) -> Tuple[float, float]:
        """
        Calculate trend of value sequence

        Args:
            values: List of values
            window: Sliding window size

        Returns:
            (Trend score, trend description)
            - Trend score: Positive value indicates downward trend, negative value indicates upward trend, larger absolute value means stronger trend
            - Trend description: Text description
        """
        if len(values) < 2:
            return 0.0, "Insufficient data"

        # Calculate overall trend (linear fit slope of last window points)
        n_points = min(window, len(values))
        if n_points < 2:
            return 0.0, "Insufficient data"
        
        last_values = values[-n_points:]
        x = np.arange(n_points)
        
        try:
            # Linear regression
            coeffs = np.polyfit(x, last_values, 1)
            slope = coeffs[0]  # Slope

            # Calculate R² value
            y_pred = coeffs[0] * x + coeffs[1]
            ss_res = np.sum((last_values - y_pred) ** 2)
            ss_tot = np.sum((last_values - np.mean(last_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Trend score = -slope * r_squared (negative slope indicates decrease)
            trend_score = -slope * r_squared

            # Description
            if trend_score > 0.1:
                trend_desc = "strong decrease"
            elif trend_score > 0.01:
                trend_desc = "decrease"
            elif trend_score > -0.01:
                trend_desc = "stable"
            elif trend_score > -0.1:
                trend_desc = "increase"
            else:
                trend_desc = "strong increase"

            return trend_score, trend_desc

        except:
            return 0.0, "Calculation failed"
    
    def check_energy_trend(self, data: Dict[str, List[float]]) -> Tuple[float, str]:
        """
        Check energy convergence trend rule

        Args:
            data: Parsed OSZICAR data

        Returns:
            (Rule score, reason description)
        """
        if not data['dE'] or len(data['dE']) < 3:
            return 0.0, "Insufficient energy change data"

        # Calculate energy change trend
        trend_score, trend_desc = self.calculate_trend(data['dE'])

        # Energy change should gradually decrease (positive trend score)
        if trend_score > 0.04:
            score = 1.0
            reason = f"Energy change shows {trend_desc} trend (trend_score={trend_score:.3f})"
        elif trend_score > 0:
            score = 0.7
            reason = f"Energy change shows slight {trend_desc} trend (trend_score={trend_score:.3f})"
        elif trend_score > -0.04:
            score = 0.3
            reason = f"Energy change trend unclear (trend_score={trend_score:.3f})"
        else:
            score = 0.0
            reason = f"Energy change shows {trend_desc} trend, may diverge (trend_score={trend_score:.3f})"

        # Check last few points
        last_dE = data['dE'][-1] if data['dE'] else 1.0
        if last_dE < 1e-4:
            score = min(score + 0.2, 1.0)
            reason += f", current dE={last_dE:.2e} is very small"
        elif last_dE > 0.1:
            score = max(score - 0.3, 0.0)
            reason += f", current dE={last_dE:.2e} is large"

        return score, reason
    
    def check_charge_density_trend(self, data: Dict[str, List[float]]) -> Tuple[float, str]:
        """
        Check charge density convergence trend rule

        Args:
            data: Parsed OSZICAR data

        Returns:
            (Rule score, reason description)
        """
        if not data['rms'] or len(data['rms']) < 3:
            return 0.0, "Insufficient charge density data"

        # Calculate charge density residual trend
        trend_score, trend_desc = self.calculate_trend(data['rms'])

        # Charge density residual should gradually decrease
        if trend_score > 0.03:
            score = 1.0
            reason = f"Charge density residual shows {trend_desc} trend (trend_score={trend_score:.3f})"
        elif trend_score > 0:
            score = 0.7
            reason = f"Charge density residual shows slight {trend_desc} trend (trend_score={trend_score:.3f})"
        elif trend_score > -0.03:
            score = 0.3
            reason = f"Charge density residual trend unclear (trend_score={trend_score:.3f})"
        else:
            score = 0.0
            reason = f"Charge density residual shows {trend_desc} trend, may diverge (trend_score={trend_score:.3f})"

        # Check last few points
        last_rms = data['rms'][-1] if data['rms'] else 1.0
        if last_rms < 1e-3:
            score = min(score + 0.2, 1.0)
            reason += f", current rms={last_rms:.2e} is very small"
        elif last_rms > 0.1:
            score = max(score - 0.3, 0.0)
            reason += f", current rms={last_rms:.2e} is large"

        return score, reason
    
    def check_oscillation(self, data: Dict[str, List[float]]) -> Tuple[float, str]:
        """
        Check oscillation behavior

        Args:
            data: Parsed OSZICAR data

        Returns:
            (Rule score, reason description)
        """
        if len(data['dE']) < 5:
            return 0.5, "Insufficient data to detect oscillation"

        # Detect oscillation in energy changes
        dE_values = data['dE'][-10:] if len(data['dE']) >= 10 else data['dE']

        # Calculate number of direction changes
        direction_changes = 0
        for i in range(1, len(dE_values) - 1):
            diff1 = dE_values[i] - dE_values[i-1]
            diff2 = dE_values[i+1] - dE_values[i]
            if diff1 * diff2 < 0:  # Direction change
                direction_changes += 1

        # Calculate oscillation ratio
        oscillation_ratio = direction_changes / max(1, len(dE_values) - 2)

        if oscillation_ratio < 0.2:
            score = 1.0
            reason = f"Slight oscillation (ratio={oscillation_ratio:.2f})"
        elif oscillation_ratio < 0.4:
            score = 0.5
            reason = f"Moderate oscillation (ratio={oscillation_ratio:.2f})"
        else:
            score = 0.0
            reason = f"Severe oscillation (ratio={oscillation_ratio:.2f})"

        return score, reason
    
    def check_convergence_threshold(self, data: Dict[str, List[float]]) -> Tuple[float, str]:
        """
        Check convergence threshold rule

        Args:
            data: Parsed OSZICAR data

        Returns:
            (Rule score, reason description)
        """
        if not data['dE'] or not data['rms']:
            return 0.0, "Insufficient data"

        last_dE = data['dE'][-1]
        last_rms = data['rms'][-1]

        # VASP common convergence criteria
        dE_threshold = 6e-3  # Energy convergence criterion
        rms_threshold = 7e-2  # Charge density convergence criterion

        # Check if convergence criteria are met
        dE_converged = last_dE < dE_threshold
        rms_converged = last_rms < rms_threshold

        if dE_converged and rms_converged:
            score = 1.0
            reason = f"Convergence criteria met (dE={last_dE:.2e}<{dE_threshold:.0e}, rms={last_rms:.2e}<{rms_threshold:.0e})"
        elif dE_converged:
            score = 0.7
            reason = f"Energy converged (dE={last_dE:.2e}<{dE_threshold:.0e}), but charge density not converged (rms={last_rms:.2e})"
        elif rms_converged:
            score = 0.7
            reason = f"Charge density converged (rms={last_rms:.2e}<{rms_threshold:.0e}), but energy not converged (dE={last_dE:.2e})"
        else:
            score = 0.3
            reason = f"Neither criterion met (dE={last_dE:.2e}, rms={last_rms:.2e})"

        return score, reason
    
    def apply_rules(self, data: Dict[str, List[float]]) -> Tuple[float, str, Dict[str, float]]:
        """
        Apply all rules and make comprehensive judgment

        Args:
            data: Parsed OSZICAR data

        Returns:
            (Comprehensive score, comprehensive reason, rule details)
        """
        if not data['dE'] or len(data['dE']) < 3:
            return 0.0, "Insufficient data, unable to judge", {}

        rule_results = {}
        total_score = 0.0
        total_weight = 0.0
        reasons = []

        for rule in self.rules:
            rule_name = rule.get('name', 'Unknown rule')
            weight = rule.get('weight', 1.0)
            check_type = rule.get('check', '')

            # Apply corresponding rule
            if check_type == 'energy_trend':
                score, reason = self.check_energy_trend(data)
            elif check_type == 'charge_density_trend':
                score, reason = self.check_charge_density_trend(data)
            elif check_type == 'oscillation_check':
                score, reason = self.check_oscillation(data)
            elif check_type == 'convergence_threshold':
                score, reason = self.check_convergence_threshold(data)
            else:
                score, reason = 0.5, "Unknown rule type"

            rule_results[rule_name] = {
                'score': score,
                'weight': weight,
                'reason': reason
            }

            total_score += score * weight
            total_weight += weight
            reasons.append(f"{rule_name}: {reason}")

        # Calculate weighted average score
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.5

        # Generate comprehensive reason
        if final_score >= 0.7:
            prediction = 1
            summary = "Very likely to converge"
        elif final_score >= 0.4:
            prediction = 1 if final_score >= 0.5 else 0
            summary = "Likely to converge" if prediction == 1 else "Likely to diverge"
        else:
            prediction = 0
            summary = "Very likely to diverge"

        final_reason = f"{summary} (score={final_score:.3f}). Details: " + "; ".join(reasons[:3])

        return final_score, final_reason, rule_results
    
    def evaluate_single_sample(self, filepath: str, true_label: int, n_steps: int = 50) -> Dict[str, Any]:
        """
        Evaluate a single sample

        Args:
            filepath: OSZICAR file path
            true_label: True label (0 or 1)
            n_steps: Number of steps to truncate

        Returns:
            Evaluation result dictionary
        """
        result = {
            'filepath': filepath,
            'true_label': true_label,
            'n_steps': n_steps,
            'prediction': None,
            'reasoning': '',
            'roc_probability': None,
            'success': False,
            'error': '',
            'api_time': 0,
            'confidence_score': None,
            'actual_steps': 0,
            'enough_steps': False,
            'rule_details': {},
            'rule_score': None
        }

        start_time = time.time()

        try:
            # Truncate OSZICAR data
            oszicar_content, actual_steps, enough_steps = self.truncate_oszicar(filepath, n_steps)
            if not oszicar_content or actual_steps < 3:
                result['error'] = f"Insufficient OSZICAR file data: {actual_steps} steps"
                return result

            # Parse OSZICAR data
            data = self.parse_oszicar_data(oszicar_content)
            if not data['dE'] or len(data['dE']) < 3:
                result['error'] = f"Insufficient parsed data: {len(data['dE'])} energy points"
                return result

            # Apply rules for judgment
            rule_score, reasoning, rule_details = self.apply_rules(data)

            # Determine prediction result (0 or 1)
            prediction = 1 if rule_score >= 0.5 else 0

            # Calculate confidence (0-9 integer)
            # Map rule score 0-1 to confidence 0-9
            confidence = int(min(9, max(0, round(rule_score * 9))))


            # Fill result
            result['success'] = True
            result['prediction'] = prediction
            result['reasoning'] = reasoning
            result['confidence_score'] = confidence
            result['rule_score'] = rule_score
            result['rule_details'] = json.dumps(rule_details, ensure_ascii=False)
            result['actual_steps'] = actual_steps
            result['enough_steps'] = enough_steps
            result['api_time'] = time.time() - start_time

        except Exception as e:
            result['error'] = f"Processing failed: {str(e)}"
            if self.debug:
                import traceback
                result['error'] += f"\n{traceback.format_exc()}"

        return result
    
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
    
    def filter_samples_for_n_steps(self, df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        """
        Filter suitable samples based on n_steps

        Args:
            df: Dataset DataFrame
            n_steps: Test step count

        Returns:
            Filtered DataFrame
        """
        if 'num_steps' not in df.columns:
            print("Warning: Dataset does not contain step count information, using file parsing method")
            return self._filter_samples_by_file_parsing(df, n_steps)

        filtered_df = df[df['num_steps'] >= n_steps].copy()

        # Verify file existence
        existing_files = []
        for _, row in filtered_df.iterrows():
            if os.path.exists(row['filepath']):
                existing_files.append(row)
            elif self.debug:
                print(f"Skipping sample {os.path.basename(row['filepath'])}: File does not exist")

        final_df = pd.DataFrame(existing_files)
        print(f"Filtered {len(final_df)} samples suitable for {n_steps}-step testing from {len(df)} samples")
        return final_df
    
    def _filter_samples_by_file_parsing(self, df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        """
        Filter samples by parsing files

        Args:
            df: Dataset DataFrame
            n_steps: Test step count

        Returns:
            Filtered DataFrame
        """
        filtered_samples = []

        for _, row in df.iterrows():
            filepath = row['filepath']
            if os.path.exists(filepath):
                # Parse actual step count
                content, actual_steps, _ = self.truncate_oszicar(filepath, n_steps)
                if actual_steps >= n_steps:
                    filtered_samples.append(row)
                elif self.debug:
                    print(f"Skipping sample {os.path.basename(filepath)}: Actual steps {actual_steps} < test steps {n_steps}")
            elif self.debug:
                print(f"Skipping sample {os.path.basename(filepath)}: File does not exist")

        filtered_df = pd.DataFrame(filtered_samples)
        print(f"Filtered {len(filtered_df)} samples suitable for {n_steps}-step testing from {len(df)} samples")
        return filtered_df
    
    def run_evaluation(self, dataset_path: str, n_steps: int = 50,
                      max_samples: int = None, output_file: str = None,
                      console_log_file: str = None) -> List[Dict[str, Any]]:
        """
        Run complete evaluation

        Args:
            dataset_path: Dataset file path
            n_steps: Number of steps to truncate
            max_samples: Maximum number of test samples
            output_file: Result output file path
            console_log_file: Console output log file path

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

        header_line = f"Starting rule-based evaluation of {total_samples} samples, steps: {n_steps}"
        table_header = f"{'#':5} {'Status':4} {'Sample':40} {'Prediction':12} {'Truth':12} {'Steps':5} {'RuleScore':10} {'Confidence':10} {'Time(s)':6}"
        separator = "-" * 103

        print(header_line)
        print(table_header)
        print(separator)

        console_log_lines.extend([header_line, table_header, separator])

        for idx, (_, row) in enumerate(df.iterrows()):
            filepath = row['filepath']
            true_label = row['label']
            actual_steps = row.get('num_steps', 'Unknown')

            result = self.evaluate_single_sample(filepath, true_label, n_steps)
            results.append(result)

            # Last part of filename
            filename = os.path.basename(filepath)
            dirname = os.path.basename(os.path.dirname(filepath))
            short_name = f"{dirname}/{filename}"

            if result['success']:
                pred_label = "Converge" if result['prediction'] == 1 else "Diverge"
                true_label_str = "Converge" if true_label == 1 else "Diverge"
                rule_score = f"{result.get('rule_score', 0):.3f}"
                confidence_score = result.get('confidence_score', 'N/A')

                # Determine if prediction is correct
                is_correct = result['prediction'] == true_label
                color_code = "\033[92m" if is_correct else "\033[91m"  # Green/Red
                reset_code = "\033[0m"
                status = "PASS" if is_correct else "FAIL"

                console_output = f"{idx+1:5d} {color_code}{status:4}{reset_code} {short_name:40} {pred_label:12} {true_label_str:12} {actual_steps:5} {rule_score:10} {confidence_score:10} {result['api_time']:6.2f}"
                log_output = f"{idx+1:5d} {status:4} {short_name:40} {pred_label:12} {true_label_str:12} {actual_steps:5} {rule_score:10} {confidence_score:10} {result['api_time']:6.2f}"

                print(console_output)
                console_log_lines.append(log_output)
            else:
                console_output = f"{idx+1:5d} FAIL {short_name:40} -            -            -     -        -        -"
                print(console_output)
                console_log_lines.append(console_output)

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
            df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
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
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in log_lines:
                    f.write(line + '\n')
            print(f"Console log saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save console log: {e}")
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics

        Args:
            results: Evaluation results list

        Returns:
            Performance metrics dictionary
        """
        successful_results = [r for r in results if r['success']]

        if not successful_results:
            return {}

        true_labels = [r['true_label'] for r in successful_results]
        predictions = [r['prediction'] for r in successful_results]

        # Calculate basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1_score': f1_score(true_labels, predictions, zero_division=0)
        }

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

        # Calculate average processing time
        process_times = [r['api_time'] for r in successful_results]
        if process_times:
            metrics['avg_process_time'] = sum(process_times) / len(process_times)
            metrics['min_process_time'] = min(process_times)
            metrics['max_process_time'] = max(process_times)
        return metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='VASP Convergence Prediction - Rule-Based Evaluation')
    parser.add_argument('--dataset', '-d', default='benchmark_dataset.csv',
                       help='Dataset file path')
    parser.add_argument('--steps', '-s', type=int, default=50,
                       help='Number of steps to truncate (50, 100, 150)')
    parser.add_argument('--max-samples', '-m', type=int, default=None,
                       help='Maximum number of test samples')
    parser.add_argument('--output', '-o', default=f'rule_based_results_{int(time.time())}.csv',
                       help='Result output file path')
    parser.add_argument('--metrics-output',
                       help='Performance metrics output file path (optional)')
    parser.add_argument('--console-log',
                       help='Console output log file path (optional)')
    parser.add_argument('--rule-config',
                       help='Rule configuration file path (JSON format, optional)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode, show detailed output')

    args = parser.parse_args()

    # Create evaluator
    evaluator = RuleBasedConvergenceEvaluator(
        rule_config=args.rule_config,
        debug=args.debug
    )

    # Run evaluation
    results = evaluator.run_evaluation(
        dataset_path=args.dataset,
        n_steps=args.steps,
        max_samples=args.max_samples,
        output_file=args.output,
        console_log_file=args.console_log
    )

    # Calculate and display performance metrics
    if results:
        metrics = evaluator.calculate_metrics(results)
        print("\n" + "=" * 60)
        print("Rule-Based Evaluation Summary Report")
        print("=" * 60)
        if metrics:
            print(f"Total samples:      {metrics['total_samples']}")
            print(f"Successful samples: {metrics['successful_samples']} ({metrics['success_rate']:.1%})")
            print(f"Accuracy:           {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
            print(f"Precision:          {metrics['precision']:.3f}")
            print(f"Recall:             {metrics['recall']:.3f}")
            print(f"F1 Score:           {metrics['f1_score']:.3f}")

            # Confusion matrix
            if 'true_positives' in metrics:
                print(f"\nConfusion Matrix:")
                print(f"  TP: {metrics['true_positives']:3d}  FP: {metrics['false_positives']:3d}")
                print(f"  FN: {metrics['false_negatives']:3d}  TN: {metrics['true_negatives']:3d}")

            # Processing performance
            if 'avg_process_time' in metrics:
                print(f"\nProcessing Performance:")
                print(f"  Average: {metrics['avg_process_time']:.4f}s")
                print(f"  Minimum: {metrics['min_process_time']:.4f}s")
                print(f"  Maximum: {metrics['max_process_time']:.4f}s")

            # Save performance metrics to CSV file
            if args.metrics_output:
                # Add evaluation info and timestamp
                metrics['method'] = 'rule_based'
                metrics['n_steps'] = args.steps
                metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(args.metrics_output, index=False, encoding='utf-8-sig')
                print(f"Performance metrics saved to: {args.metrics_output}")

        else:
            print("No successful samples")


if __name__ == "__main__":
    main()
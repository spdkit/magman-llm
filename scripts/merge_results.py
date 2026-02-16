#!/usr/bin/env python3
"""
Results Merge Tool - Run in IPython
Merge multiple result files generated from parallel evaluation
"""

import pandas as pd
import glob
import argparse

def merge_results(pattern='llm_results_*.csv', output_file=None):
    """
    Merge parallel evaluation result files

    Args:
        pattern: Result file matching pattern
        output_file: Output file path
    """
    # Find all result files
    result_files = glob.glob(pattern)

    if not result_files:
        print(f"No result files found matching {pattern}")
        return

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f}")

    # Merge all results
    print("\nMerging result files...")
    all_dfs = []

    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            print(f"Read {file_path}: {len(df)} samples")
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    if not all_dfs:
        print("No result files successfully read")
        return
    
    # Merge DataFrames
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Set output filename
    if output_file is None:
        # Extract timestamp from first file
        first_file = result_files[0]
        if 'llm_results_' in first_file and '.csv' in first_file:
            timestamp = first_file.split('llm_results_')[1].split('.csv')[0]
            output_file = f'final_results_{timestamp}.csv'
        else:
            output_file = 'final_results_combined.csv'

    # Save merged results
    df_combined.to_csv(output_file, index=False)

    print(f"\nMerge completed!")
    print(f"Total samples: {len(df_combined)}")
    print(f"Results saved to: {output_file}")

    # Display basic statistics
    successful_samples = len(df_combined[df_combined['success'] == True])
    success_rate = successful_samples / len(df_combined) if len(df_combined) > 0 else 0

    print(f"Successfully evaluated samples: {successful_samples} ({success_rate:.1%})")

    if successful_samples > 0:
        from sklearn.metrics import accuracy_score
        true_labels = df_combined[df_combined['success'] == True]['true_label']
        predictions = df_combined[df_combined['success'] == True]['prediction']
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {accuracy:.3f}")

def merge_results_by_steps(n_steps, output_file=None):
    """
    Merge result files by step count
    """
    pattern = f'llm_results_*_steps{n_steps}.csv'
    if output_file is None:
        output_file = f'final_results_steps{n_steps}.csv'

    merge_results(pattern, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge parallel evaluation result files')
    parser.add_argument('--pattern', default='llm_results_*.csv', help='Result file matching pattern')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--steps', type=int, help='Merge by step count (automatically matches corresponding files when specified)')
    
    args = parser.parse_args()
    
    if args.steps:
        merge_results_by_steps(args.steps, args.output)
    else:
        merge_results(args.pattern, args.output)
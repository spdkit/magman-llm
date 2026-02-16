#!/usr/bin/env python3
"""
Dataset Split Tool - Run in IPython
Filter benchmark_dataset.csv by step count and split into multiple sub-files for parallel evaluation
"""

import pandas as pd
import argparse

def split_dataset(dataset_path='benchmark_dataset.csv', n_steps=100, num_parts=4):
    """
    Split dataset for parallel evaluation

    Args:
        dataset_path: Dataset file path
        n_steps: Test step count
        num_parts: Number of parts to split into
    """
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Filter samples suitable for n_steps testing
    print(f"Filtering samples with steps >= {n_steps}...")
    df_filtered = df[df['num_steps'] >= n_steps]

    # Sort by filepath to ensure determinism
    df_filtered = df_filtered.sort_values('filepath').reset_index(drop=True)

    total_samples = len(df_filtered)
    print(f"Found {total_samples} samples suitable for {n_steps} step testing")

    if total_samples == 0:
        print("No suitable samples found, please check step count setting")
        return

    # Calculate samples per part
    samples_per_part = total_samples // num_parts
    remainder = total_samples % num_parts

    # Split into multiple files
    print(f"Splitting into {num_parts} parts...")

    for i in range(num_parts):
        # Calculate sample range for current part (handle remainder distribution)
        start_idx = i * samples_per_part + min(i, remainder)
        end_idx = start_idx + samples_per_part + (1 if i < remainder else 0)

        part_df = df_filtered.iloc[start_idx:end_idx]
        output_file = f'dataset_steps{n_steps}_part{i}.csv'
        part_df.to_csv(output_file, index=False)

        print(f"Created {output_file}: {len(part_df)} samples")

    print(f"\nSplit completed! Total {total_samples} samples divided into {num_parts} parts")
    print("\nYou can now run in each terminal:")
    for i in range(num_parts):
        print(f"python evaluate_llm_convergence.py --dataset dataset_steps{n_steps}_part{i}.csv --steps {n_steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset for parallel evaluation')
    parser.add_argument('--dataset', default='benchmark_dataset.csv', help='Dataset file path')
    parser.add_argument('--steps', type=int, default=100, help='Test step count')
    parser.add_argument('--parts', type=int, default=4, help='Number of parts to split into')
    
    args = parser.parse_args()
    
    split_dataset(args.dataset, args.steps, args.parts)
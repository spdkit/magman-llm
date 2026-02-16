#!/usr/bin/env python3
"""
Count and list CSV files in the project
"""

import os
import glob
from pathlib import Path
from typing import Dict, List
import pandas as pd

def find_files_by_pattern(pattern: str) -> List[str]:
    """Find all files matching the pattern"""
    return glob.glob(pattern, recursive=True)

def count_csv_files() -> Dict[str, Dict]:
    """Count CSV files and collect detailed information"""

    # Find all CSV files
    csv_files = find_files_by_pattern('**/*.csv')
    # Filter out hidden files
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('.')]
    
    result = {
        'csv_files': {
            'count': len(csv_files),
            'files': sorted(csv_files),
            'size_info': {}
        }
    }

    # Add file size information
    for file_path in csv_files:
        try:
            size = os.path.getsize(file_path)
            result['csv_files']['size_info'][file_path] = f"{size / 1024:.1f} KB"
        except OSError:
            result['csv_files']['size_info'][file_path] = "N/A"

    return result

def analyze_csv_files(csv_files: List[str]) -> Dict:
    """Analyze CSV file content structure"""
    analysis = {
        'total_files': len(csv_files),
        'file_types': {},
        'sample_counts': {},
        'columns_analysis': {}
    }
    
    for csv_file in csv_files:
        try:
            # Read basic CSV file information
            df = pd.read_csv(csv_file, nrows=1)
            file_name = os.path.basename(csv_file)

            # Count file types (based on filename patterns)
            if 'enhanced_results' in file_name:
                file_type = 'enhanced_results'
            elif 'roc_analysis' in file_name:
                file_type = 'roc_analysis'
            else:
                file_type = 'raw_results'

            analysis['file_types'][file_type] = analysis['file_types'].get(file_type, 0) + 1

            # Record column information
            analysis['columns_analysis'][csv_file] = list(df.columns)

            # Try to get sample count (without reading entire file)
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header row
                analysis['sample_counts'][csv_file] = line_count
            except:
                analysis['sample_counts'][csv_file] = 'N/A'

        except Exception as e:
            print(f"Cannot analyze file {csv_file}: {e}")
    
    return analysis

def print_statistics(results: Dict):
    """Print statistics results"""
    print("=" * 60)
    print("CSV File Statistics Report")
    print("=" * 60)

    csv_data = results['csv_files']
    print(f"Total CSV files: {csv_data['count']}")
    print()

    if csv_data['files']:
        print("CSV file list:")
        for i, file_path in enumerate(csv_data['files'], 1):
            size_info = csv_data['size_info'].get(file_path, 'N/A')
            print(f"  {i:2d}. {os.path.basename(file_path)} ({size_info})")
            print(f"      Path: {file_path}")
        print()

        # Detailed CSV file analysis
        csv_analysis = analyze_csv_files(csv_data['files'])
        print("Detailed CSV file analysis:")
        print(f"  Total CSV files: {csv_analysis['total_files']}")
        print("  File type distribution:")
        for file_type, count in csv_analysis['file_types'].items():
            print(f"    {file_type}: {count}")

        # Display sample count statistics
        sample_counts = [count for count in csv_analysis['sample_counts'].values()
                        if isinstance(count, int)]
        if sample_counts:
            print(f"  Sample count range: {min(sample_counts)} - {max(sample_counts)}")
            print(f"  Average sample count: {sum(sample_counts) / len(sample_counts):.1f}")
        print()
    else:
        print("No CSV files found")
        print()

def main():
    """Main function"""
    print("Scanning CSV files...")

    # Get current project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    results = count_csv_files()
    print_statistics(results)

    # Save detailed report to file
    report_file = "csv_statistics_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print_statistics(results)
        sys.stdout = original_stdout

    print(f"Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
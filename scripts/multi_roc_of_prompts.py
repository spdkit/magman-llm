import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Dict, List, Optional
import argparse
# Example command: python scripts/multi2.py --folders 02_50steps_n_samples/50samples_qwen3-14b_analysis 02_50steps_n_samples/80samples_qwen3-14b_analysis 02_50steps_n_samples/80samples_qwen3-14b_analysis 02_50steps_n_samples/100samples_qwen3-14b_analysis 02_50steps_n_samples/120samples_qwen3-14b_analysis 02_50steps_n_samples/150samples_qwen3-14b_analysis --output pictures/n_example --name summary_analysis
class MultiModelAnalyzer:
    """
    Multi-model analyzer for reading analysis results from multiple folders and generating comprehensive charts
    """

    def __init__(self, folder_paths: List[str], output_dir: str = "combined_results", custom_name: str = None):
        """
        Initialize analyzer

        Args:
            folder_paths: List of folder paths containing analysis results
            output_dir: Output directory path
            custom_name: Custom output folder name
        """
        self.folder_paths = folder_paths

        # Handle output directory
        if custom_name:
            # If custom name specified, create subfolder with that name
            self.output_dir = os.path.join(output_dir, custom_name)
        else:
            # Otherwise use default name
            self.output_dir = os.path.join(output_dir, "multi_model_analysis")

        self.all_data = {}  # Store all model data
        self.model_names = []  # Model name list
        self.ap_scores = {}  # Store AP scores
        self.auc_scores = {}  # Store AUC scores

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Read all data
        self._load_all_data()
    
    def _load_all_data(self):
        """Read data from all specified folders"""
        print("📁 Starting to read multi-folder data...")

        for i, folder_path in enumerate(self.folder_paths):
            # Generate model name (use folder name or index)
            model_name = os.path.basename(folder_path.rstrip('/')) if os.path.basename(folder_path) else f"model_{i+1}"
            self.model_names.append(model_name)

            try:
                # Read analysis report
                analysis_report_path = os.path.join(folder_path, "analysis_report.json")
                with open(analysis_report_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)

                # Read processed data
                processed_data_path = os.path.join(folder_path, "processed_data.json")
                with open(processed_data_path, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)

                # Store data
                self.all_data[model_name] = {
                    'analysis_report': analysis_data,
                    'processed_data': processed_data
                }

                print(f"✅ Successfully loaded data for model {model_name}")
                print(f"   Sample count: {processed_data.get('total_samples', 'N/A')}")

            except FileNotFoundError as e:
                print(f"❌ File not found in folder {folder_path}: {e}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON file: {e}")
            except Exception as e:
                print(f"❌ Error reading data from folder {folder_path}: {e}")
    
    def _calculate_ap_score(self, model_name: str, data: dict) -> float:
        """
        Calculate AP score (Average Precision)

        AP (Average Precision) is the area under the PR curve, an important metric for evaluating classification model performance.
        Similar to AUC, but AP focuses more on positive class identification performance.

        Args:
            model_name: Model name
            data: Model data

        Returns:
            AP score
        """
        try:
            processed_data = data['processed_data']

            # Get true labels and confidence scores
            true_labels = np.array(processed_data['true_labels'])

            # Try to get confidence scores
            if 'roc_probabilities' in processed_data:
                probabilities = np.array(processed_data['roc_probabilities'])
            elif 'confidence_scores' in processed_data:
                confidence_scores = np.array(processed_data['confidence_scores'])
                probabilities = self._normalize_confidence_scores(confidence_scores)
            else:
                print(f"⚠️ Model {model_name} has no confidence data, cannot calculate AP")
                return 0.0

            # Calculate AP score
            ap_score = average_precision_score(true_labels, probabilities)
            return ap_score

        except Exception as e:
            print(f"❌ Error calculating AP score for model {model_name}: {e}")
            return 0.0

    def _calculate_auc_score(self, model_name: str, data: dict) -> float:
        """
        Calculate AUC score (Area Under ROC Curve)

        Args:
            model_name: Model name
            data: Model data

        Returns:
            AUC score
        """
        try:
            processed_data = data['processed_data']

            # Get true labels and confidence scores
            true_labels = np.array(processed_data['true_labels'])

            # Try to get confidence scores
            if 'roc_probabilities' in processed_data:
                probabilities = np.array(processed_data['roc_probabilities'])
            elif 'confidence_scores' in processed_data:
                confidence_scores = np.array(processed_data['confidence_scores'])
                probabilities = self._normalize_confidence_scores(confidence_scores)
            else:
                print(f"⚠️ Model {model_name} has no confidence data, cannot calculate AUC")
                return 0.0

            # Calculate AUC score
            fpr, tpr, _ = roc_curve(true_labels, probabilities)
            auc_score = auc(fpr, tpr)
            return auc_score

        except Exception as e:
            print(f"❌ Error calculating AUC score for model {model_name}: {e}")
            return 0.0
    
    def plot_combined_roc_curves(self):
        """
        Plot ROC curves for all models on the same chart
        """
        if not self.all_data:
            print("❌ No data available, cannot plot ROC curves")
            return

        plt.figure(figsize=(12, 10))

        # Color list, assign different colors to different models
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # Plot ROC curve for each model
        for i, (model_name, data) in enumerate(self.all_data.items()):
            try:
                processed_data = data['processed_data']

                # Get true labels and confidence scores
                true_labels = np.array(processed_data['true_labels'])

                # Try to get confidence scores
                if 'roc_probabilities' in processed_data:
                    probabilities = np.array(processed_data['roc_probabilities'])
                elif 'confidence_scores' in processed_data:
                    confidence_scores = np.array(processed_data['confidence_scores'])
                    probabilities = self._normalize_confidence_scores(confidence_scores)
                else:
                    print(f"⚠️ Model {model_name} has no confidence data, skipping")
                    continue

                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
                roc_auc = auc(fpr, tpr)
                self.auc_scores[model_name] = roc_auc

                # Select color
                color = colors[i % len(colors)]

                # Plot ROC curve
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')

                print(f"✅ Processed model {model_name}: AUC = {roc_auc:.3f}")

            except Exception as e:
                print(f"❌ Error processing model {model_name}: {e}")
                continue

        # Plot random classifier baseline
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier', alpha=0.5)

        # Set chart properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=20)
        plt.ylabel('True Positive Rate (TPR)', fontsize=20)
        plt.title('Combined ROC Curves for Models', fontsize=25, fontweight='bold')
        plt.legend(loc="lower right", fontsize=14)
        plt.grid(True, alpha=0.3)

        # Save image
        output_file = os.path.join(self.output_dir, 'combined_roc_curves.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"🎉 Combined ROC curves saved: {output_file}")

        return self.auc_scores
    
    def plot_combined_pr_curves(self):
        """
        Plot PR curves for all models on the same chart
        """
        if not self.all_data:
            print("❌ No data available, cannot plot PR curves")
            return

        plt.figure(figsize=(12, 10))

        # Color list
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # Plot PR curve for each model
        for i, (model_name, data) in enumerate(self.all_data.items()):
            try:
                processed_data = data['processed_data']

                # Get true labels and confidence scores
                true_labels = np.array(processed_data['true_labels'])

                # Try to get confidence scores
                if 'roc_probabilities' in processed_data:
                    probabilities = np.array(processed_data['roc_probabilities'])
                elif 'confidence_scores' in processed_data:
                    confidence_scores = np.array(processed_data['confidence_scores'])
                    probabilities = self._normalize_confidence_scores(confidence_scores)
                else:
                    print(f"⚠️ Model {model_name} has no confidence data, skipping")
                    continue

                # Calculate PR curve and AP score
                precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
                avg_precision = average_precision_score(true_labels, probabilities)
                self.ap_scores[model_name] = avg_precision

                # Select color
                color = colors[i % len(colors)]

                # Plot PR curve
                plt.plot(recall, precision, color=color, lw=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')

                print(f"✅ Processed model {model_name}: AP = {avg_precision:.3f}")

            except Exception as e:
                print(f"❌ Error processing model {model_name}: {e}")
                continue

        # Calculate and plot random classifier baseline (positive ratio)
        if self.all_data:
            # Use first model's data to calculate positive ratio
            first_model_data = list(self.all_data.values())[0]['processed_data']
            true_labels = np.array(first_model_data['true_labels'])
            positive_ratio = np.mean(true_labels)

            plt.axhline(y=positive_ratio, color='red', linestyle='--',
                       label=f'Random classifier (AP = {positive_ratio:.3f})', alpha=0.5)

        # Set chart properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.title('Combined Precision-Recall Curves for Models', fontsize=25, fontweight='bold')
        plt.legend(loc="upper right", fontsize=14)
        plt.grid(True, alpha=0.3)

        # Save image
        output_file = os.path.join(self.output_dir, 'combined_pr_curves.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"🎉 Combined PR curves saved: {output_file}")

        return self.ap_scores
    
    def _normalize_confidence_scores(self, confidence_scores: np.ndarray) -> np.ndarray:
        """
        Normalize confidence scores to 0-1 range

        Args:
            confidence_scores: Original confidence scores

        Returns:
            Normalized probability values
        """
        scores = confidence_scores.astype(float)

        # If already in 0-1 range, return directly
        if scores.max() <= 1.0:
            return scores

        # If in 0-100 range, divide by 100
        elif scores.max() <= 100.0:
            return scores / 100.0

        # Other cases, try to find appropriate scaling factor
        else:
            max_val = scores.max()
            if max_val <= 10.0:
                return scores / 10.0
            else:
                # Default: divide by max value
                return scores / max_val
    
    def generate_summary_report(self):
        """
        Generate comprehensive performance report including AP and AUC data

        AP (Average Precision): Average precision, area under PR curve
        - Range: 0-1, closer to 1 indicates better performance
        - Characteristic: More sensitive to imbalanced datasets
        - Relationship with AUC: Both are areas under curves, but AP focuses on PR curve, AUC focuses on ROC curve
        """
        if not self.all_data:
            print("❌ No data available, cannot generate report")
            return

        print("\n📊 Model Performance Summary Report")
        print("=" * 60)

        summary_data = []

        for model_name, data in self.all_data.items():
            try:
                analysis_report = data['analysis_report']
                model_results = analysis_report['model_results'].get('single_model', {})

                # Calculate AP score (if not already calculated)
                if model_name not in self.ap_scores:
                    self.ap_scores[model_name] = self._calculate_ap_score(model_name, data)

                # Calculate AUC score (if not already calculated)
                if model_name not in self.auc_scores:
                    self.auc_scores[model_name] = self._calculate_auc_score(model_name, data)

                summary_data.append({
                    'Model': model_name,
                    'Accuracy': f"{model_results.get('accuracy', 0):.4f}",
                    'Precision': f"{model_results.get('precision', 0):.4f}",
                    'Recall': f"{model_results.get('recall', 0):.4f}",
                    'F1-Score': f"{model_results.get('f1_score', 0):.4f}",
                    'AUC': f"{self.auc_scores.get(model_name, 0):.4f}",
                    'AP': f"{self.ap_scores.get(model_name, 0):.4f}",  # Added AP column
                    'Specificity': f"{model_results.get('specificity', 0):.4f}",
                    'Balanced_Accuracy': f"{model_results.get('balanced_accuracy', 0):.4f}",
                    'Samples': analysis_report['data_info'].get('total_samples', 'N/A')
                })

            except Exception as e:
                print(f"❌ Error generating report for model {model_name}: {e}")
                continue

        # Create DataFrame and display
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))

        # Save to CSV file
        csv_file = os.path.join(self.output_dir, 'model_performance_summary.csv')
        df_summary.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 Performance summary table saved: {csv_file}")

        # Also save detailed performance metrics to JSON
        json_file = os.path.join(self.output_dir, 'detailed_performance_metrics.json')
        detailed_metrics = {
            'summary': summary_data,
            'ap_scores': self.ap_scores,
            'auc_scores': self.auc_scores
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed performance metrics saved: {json_file}")

        return df_summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-model performance analysis tool')
    parser.add_argument('--folders', nargs='+', required=True,
                       help='List of folder paths containing analysis results')
    parser.add_argument('--output', '-o', default='./results',
                       help='Output directory path (default: ./results)')
    parser.add_argument('--name', '-n',
                       help='Custom output folder name')

    return parser.parse_args()

# Usage example and auto-detection
def main():
    """
    Main function - demonstrates how to use multi-model analyzer
    """
    # Parse command line arguments
    args = parse_arguments()

    # Use command line arguments
    folder_paths = args.folders
    output_dir = args.output
    custom_name = args.name

    # Check if folders exist
    valid_folders = []
    for folder in folder_paths:
        if os.path.exists(folder):
            valid_folders.append(folder)
        else:
            print(f"⚠️ Folder does not exist: {folder}")

    if not valid_folders:
        print("❌ No valid folders found, please check paths")
        print("\n💡 Usage:")
        print("python multi_model_analyzer.py --folders folder1 folder2 folder3 --output ./results --name my_analysis")
        return

    # Create analyzer instance
    print("🚀 Initializing multi-model analyzer...")
    print(f"Input folders: {valid_folders}")
    print(f"Output path: {output_dir}")
    if custom_name:
        print(f"Custom name: {custom_name}")

    analyzer = MultiModelAnalyzer(valid_folders, output_dir, custom_name)

    # Generate comprehensive report
    print("\n📈 Generating performance summary report...")
    analyzer.generate_summary_report()

    # Plot combined ROC curves
    print("\n🎨 Plotting combined ROC curves...")
    auc_scores = analyzer.plot_combined_roc_curves()

    # Plot combined PR curves
    print("\n🎨 Plotting combined PR curves...")
    ap_scores = analyzer.plot_combined_pr_curves()

    # Display final output information
    print(f"\n🎉 All tasks completed!")
    print(f"📁 Results saved in: {analyzer.output_dir}")
    print(f"📊 Includes files:")
    print(f"   - model_performance_summary.csv (performance summary table)")
    print(f"   - detailed_performance_metrics.json (detailed metrics)")
    print(f"   - combined_roc_curves.png (combined ROC curves)")
    print(f"   - combined_pr_curves.png (combined PR curves)")

# Auto-detect and run
if __name__ == "__main__":
    # Check if necessary libraries are installed
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        print("✅ All dependency libraries installed")

        # Run main function
        main()

    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Please install required libraries: pip install matplotlib pandas numpy scikit-learn")

    except Exception as e:
        print(f"❌ Error occurred during execution: {e}")
        print("Please check folder paths and file formats are correct")
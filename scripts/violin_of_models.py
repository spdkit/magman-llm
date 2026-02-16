import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import argparse
import seaborn as sns
import matplotlib as mpl

# Set global font
plt.rcParams['font.family'] = 'DejaVu Sans'  # Clearer font

class ConfidenceDistributionPlotter:
    def __init__(self, data_folder: str = ".", output_dir: str = "output"):
        """
        Initialize confidence distribution plotter

        Args:
            data_folder: Folder path containing CSV files
            output_dir: Output directory for images
        """
        self.data_folder = data_folder
        self.output_dir = output_dir
        self.df = None  # Store all data
        self.models = []  # Store model name list

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, csv_files: List[str] = None):
        """
        Load CSV file data

        Args:
            csv_files: List of CSV file names to load, if None load all CSV files in folder
        """
        all_data = []

        if csv_files is None:
            # Load all CSV files in folder
            csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]

        for csv_file in csv_files:
            # Handle file path - use full path if file not in current directory
            if os.path.isabs(csv_file):
                file_path = csv_file
            elif os.path.exists(csv_file):
                file_path = csv_file
            else:
                file_path = os.path.join(self.data_folder, csv_file)

            if os.path.exists(file_path):
                try:
                    # Read CSV file
                    df_temp = pd.read_csv(file_path)

                    # Extract model name from filename (remove path and .csv suffix)
                    base_name = os.path.basename(file_path)
                    model_name = base_name.replace('.csv', '')

                    # Remove version info like _v10 from model name
                    model_name = self._clean_model_name(model_name)

                    # Add model name column
                    df_temp['model'] = model_name

                    # Ensure confidence_score column is numeric type
                    if 'confidence_score' in df_temp.columns:
                        df_temp['confidence_score'] = pd.to_numeric(df_temp['confidence_score'], errors='coerce')

                    all_data.append(df_temp)
                    self.models.append(model_name)

                    print(f"✅ Successfully loaded data for model {model_name}, {len(df_temp)} records")

                except Exception as e:
                    print(f"❌ Error loading file {file_path}: {e}")
            else:
                print(f"⚠️ File does not exist: {file_path}")

        if all_data:
            # Merge all data
            self.df = pd.concat(all_data, ignore_index=True)
            print(f"📊 Total loaded {len(self.df)} records, containing {len(self.models)} models")

            # Check confidence score data
            if 'confidence_score' in self.df.columns:
                print(f"📈 Confidence score statistics:")
                print(f"   Data type: {self.df['confidence_score'].dtype}")
                print(f"   Valid value count: {self.df['confidence_score'].notna().sum()}")
                print(f"   Value range: {self.df['confidence_score'].min()} - {self.df['confidence_score'].max()}")
        else:
            print("❌ No data loaded successfully")
    
    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name, remove version info like _v10"""
        # Remove common version info
        patterns = ['_v10', '_v20', '_v30', '-v10', '-v20', '-v30']
        for pattern in patterns:
            model_name = model_name.replace(pattern, '')
        return model_name

    def _truncate_model_name(self, model_name: str, max_length: int = 25) -> str:
        """Truncate model name to make it readable"""
        if len(model_name) <= max_length:
            return model_name

        # Try to truncate in the middle
        half = max_length // 2
        return model_name[:half-3] + "..." + model_name[-half:]
    
    def plot_single_confidence_distribution(self, model_name: str = None):
        """
        Plot confidence score distribution for a single model

        Args:
            model_name: Model name, if None use first model
        """
        if self.df is None:
            print("❌ Please load data first")
            return

        if model_name is None:
            model_name = self.models[0]

        # Filter data for specified model
        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        # Check if confidence score column exists
        if 'confidence_score' not in model_df.columns:
            print(f"⚠️ Model {model_name} has no confidence score data")
            return

        # Count confidence score distribution (0-9)
        confidence_scores = model_df['confidence_score'].value_counts().sort_index()

        # Ensure all scores 0-9 are included (even if some scores have no samples)
        full_range = pd.Series(index=range(0, 10), data=0)
        confidence_scores = confidence_scores.reindex(full_range.index, fill_value=0)

        # Plot confidence distribution
        plt.figure(figsize=(12, 6))

        # Use viridis color scheme
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(confidence_scores)))
        bars = plt.bar(confidence_scores.index, confidence_scores.values,
                      color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)

        plt.xlabel('Confidence Score (0-9)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
        plt.title(f'{model_name} - Confidence Score Distribution',
                 fontsize=25, fontweight='bold', pad=20)
        plt.xticks(range(0, 10), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add background color
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels on bars with samples
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save image - use safe filename
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(self.output_dir, f'confidence_distribution_{safe_model_name}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Single model confidence distribution plot saved: {output_file}")

        # Print distribution statistics
        print(f"Confidence score distribution for model {model_name}:")
        for score in range(0, 10):
            count = confidence_scores.get(score, 0)
            print(f"   Confidence {score}: {count} samples")

        return confidence_scores.to_dict()
    
    def plot_multiple_confidence_distributions(self):
        """
        Plot confidence distribution summary chart for multiple models
        Plot all models' confidence distributions on the same chart for comparison
        """
        if self.df is None:
            print("❌ Please load data first")
            return

        if len(self.models) == 0:
            print("❌ No model data available")
            return

        # Create figure
        plt.figure(figsize=(14, 8))

        # Use viridis color scheme
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(self.models)))

        # Store distribution data for all models
        all_distributions = {}

        # Plot confidence distribution for each model
        for i, model_name in enumerate(self.models):
            # Filter data for current model
            model_df = self.df[self.df['model'] == model_name]

            if 'confidence_score' not in model_df.columns:
                print(f"⚠️ Model {model_name} has no confidence score data, skipping")
                continue

            # Count confidence score distribution
            confidence_scores = model_df['confidence_score'].value_counts().sort_index()

            # Ensure all scores 0-9 are included
            full_range = pd.Series(index=range(0, 10), data=0)
            confidence_scores = confidence_scores.reindex(full_range.index, fill_value=0)

            # Store distribution data
            all_distributions[model_name] = confidence_scores

            # Plot line chart
            plt.plot(confidence_scores.index, confidence_scores.values,
                    marker='o', linewidth=2.5, markersize=8,
                    color=colors[i], label=self._truncate_model_name(model_name),
                    alpha=0.9, markeredgecolor='white', markeredgewidth=1)

        # Set chart properties
        plt.xlabel('Confidence Score (0-9)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
        plt.title('Multiple Models - Confidence Score Distribution Comparison',
                 fontsize=25, fontweight='bold', pad=20)
        plt.xticks(range(0, 10), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')

        # Add background color
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')

        # Legend on the right side
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11,
                  frameon=True, fancybox=True, shadow=True)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save summary chart
        output_file = os.path.join(self.output_dir, 'confidence_distribution_comparison_line.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Multi-model confidence distribution line chart saved: {output_file}")

        # Print distribution statistics for all models
        print("\nConfidence score distribution statistics for all models:")
        for model_name, distribution in all_distributions.items():
            print(f"\nModel {model_name}:")
            total_samples = distribution.sum()
            for score in range(0, 10):
                count = distribution.get(score, 0)
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                print(f"   Confidence {score}: {count} samples ({percentage:.1f}%)")

        return all_distributions
    
    def plot_grouped_bar_chart(self):
        """
        Plot grouped bar chart, comparing confidence distributions of all models together
        """
        if self.df is None:
            print("❌ Please load data first")
            return

        if len(self.models) == 0:
            print("❌ No model data available")
            return

        # Prepare data
        grouped_data = {}
        for model_name in self.models:
            model_df = self.df[self.df['model'] == model_name]
            if 'confidence_score' not in model_df.columns:
                continue

            # Count confidence score distribution
            confidence_scores = model_df['confidence_score'].value_counts().sort_index()

            # Ensure all scores 0-9 are included
            full_range = pd.Series(index=range(0, 10), data=0)
            confidence_scores = confidence_scores.reindex(full_range.index, fill_value=0)

            grouped_data[model_name] = confidence_scores

        if not grouped_data:
            print("❌ No confidence data available")
            return

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set bar chart parameters
        n_models = len(grouped_data)
        n_scores = 10  # 0-9
        bar_width = 0.8 / n_models  # Dynamically adjust bar width
        x = np.arange(n_scores)

        # Use viridis color scheme
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_models))

        # Plot bar chart for each model
        for i, (model_name, distribution) in enumerate(grouped_data.items()):
            bars = ax.bar(x + i * bar_width, distribution.values,
                         bar_width, label=self._truncate_model_name(model_name),
                         color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

        # Set chart properties
        ax.set_xlabel('Confidence Score (0-9)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
        ax.set_title('Multiple Models - Confidence Score Distribution (Grouped Bar Chart)',
                    fontsize=25, fontweight='bold', pad=20)
        ax.set_xticks(x + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(range(0, 10), fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Add background color
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11,
                 frameon=True, fancybox=True, shadow=True)

        # Save grouped bar chart
        output_file = os.path.join(self.output_dir, 'confidence_distribution_grouped_bar.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Multi-model confidence distribution grouped bar chart saved: {output_file}")

        return grouped_data
    
    def plot_box_plot(self):
        """
        Plot box plot showing confidence score distribution for each model
        """
        if self.df is None:
            print("❌ Please load data first")
            return

        if len(self.models) == 0:
            print("❌ No model data available")
            return

        # Prepare data
        box_data = []
        model_names = []

        print("\n📊 Box plot data preparation:")
        for model_name in self.models:
            model_df = self.df[self.df['model'] == model_name]
            print(f"  Model '{model_name}': {len(model_df)} records")

            if 'confidence_score' not in model_df.columns:
                print(f"  ⚠️ Model {model_name} has no confidence score column")
                continue

            # Get confidence scores and ensure numeric type
            confidence_scores = model_df['confidence_score']

            # Convert to numeric type, handle possible string types
            try:
                confidence_scores = pd.to_numeric(confidence_scores, errors='coerce')
            except Exception as e:
                print(f"    ❌ Error converting confidence scores to numeric type: {e}")
                continue

            # Remove NaN values
            valid_scores = confidence_scores.dropna()
            print(f"    Valid confidence score count: {len(valid_scores)}")

            if len(valid_scores) > 0:
                box_data.append(valid_scores)
                model_names.append(model_name)
                print(f"    ✅ Successfully added data for model {model_name}")
            else:
                print(f"    ⚠️ Model {model_name} has no valid confidence score data")

        if not box_data:
            print("❌ No confidence data available")
            return

        print(f"\n📈 Preparing to plot box plot with data from {len(box_data)} models")

        # Create box plot
        plt.figure(figsize=(14, 8))

        # Use Set3 color scheme (box plot colors)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))

        # Plot box plot
        box_plot = plt.boxplot(box_data, labels=[self._truncate_model_name(name) for name in model_names],
                              patch_artist=True, showfliers=False, medianprops=dict(color='white', linewidth=2))

        # Set box plot colors and styles
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        # Set whisker colors and widths
        for whisker in box_plot['whiskers']:
            whisker.set(color='black', linewidth=1.5)

        for cap in box_plot['caps']:
            cap.set(color='black', linewidth=1.5)

        # Set chart properties
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence Score', fontsize=14, fontweight='bold')
        plt.title('Multiple Models - Confidence Score Distribution (Box Plot)',
                 fontsize=25, fontweight='bold', pad=20)

        # Rotate X-axis labels 45 degrees
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=12)

        # Add background color and grid
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add overall mean reference line
        all_scores = np.concatenate(box_data)
        overall_mean = np.mean(all_scores)
        plt.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Overall Mean: {overall_mean:.2f}')

        # Add legend
        plt.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)

        # Adjust layout
        plt.tight_layout()

        # Save box plot
        output_file = os.path.join(self.output_dir, 'confidence_distribution_box_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Multi-model confidence distribution box plot saved: {output_file}")

        # Print statistics
        print("\nConfidence score statistics for each model:")
        for i, model_name in enumerate(model_names):
            scores = box_data[i]
            print(f"\nModel {model_name}:")
            print(f"  Mean: {scores.mean():.2f}")
            print(f"  Median: {scores.median():.2f}")
            print(f"  Std Dev: {scores.std():.2f}")
            print(f"  Min: {scores.min():.2f}")
            print(f"  Max: {scores.max():.2f}")
            print(f"  Sample count: {len(scores)}")

        return {model: data for model, data in zip(model_names, box_data)}

    def plot_violin_plot(self):
        """
        Plot violin chart showing confidence score distribution density for each model
        """
        if self.df is None:
            print("❌ Please load data first")
            return

        if len(self.models) == 0:
            print("❌ No model data available")
            return

        # Prepare data
        violin_data = []
        violin_labels = []

        print("\n🎻 Violin plot data preparation:")
        for model_name in self.models:
            model_df = self.df[self.df['model'] == model_name]
            print(f"  Model '{model_name}': {len(model_df)} records")

            if 'confidence_score' not in model_df.columns:
                print(f"  ⚠️ Model {model_name} has no confidence score column")
                continue

            # Get confidence scores and ensure numeric type
            confidence_scores = model_df['confidence_score']

            # Convert to numeric type, handle possible string types
            try:
                confidence_scores = pd.to_numeric(confidence_scores, errors='coerce')
            except Exception as e:
                print(f"    ❌ Error converting confidence scores to numeric type: {e}")
                continue

            # Remove NaN values
            valid_scores = confidence_scores.dropna()
            print(f"    Valid confidence score count: {len(valid_scores)}")

            if len(valid_scores) > 0:
                violin_data.append(valid_scores)
                violin_labels.append(model_name)
                print(f"    ✅ Successfully added data for model {model_name}")
            else:
                print(f"    ⚠️ Model {model_name} has no valid confidence score data")

        if not violin_data:
            print("❌ No confidence data available")
            return

        print(f"\n📈 Preparing to plot violin chart with data from {len(violin_data)} models")

        # Create violin plot
        plt.figure(figsize=(16, 10))

        # Prepare long-format data suitable for seaborn
        plot_data = []
        for i, (scores, model_name) in enumerate(zip(violin_data, violin_labels)):
            for score in scores:
                plot_data.append({'model': model_name, 'confidence_score': score})

        plot_df = pd.DataFrame(plot_data)

        # Truncate model names for display
        plot_df['display_model'] = plot_df['model'].apply(self._truncate_model_name)

        # Use Set3 color scheme (box plot colors)
        palette = plt.cm.Set3(np.linspace(0, 1, len(violin_labels)))

        # Plot violin chart - set same width ratio
        ax = sns.violinplot(data=plot_df, x='display_model', y='confidence_score',
                           palette=palette, inner=None, cut=0, bw=0.15,
                           linewidth=1.5, saturation=0.8, scale='width')

        # Set chart properties
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence Score', fontsize=14, fontweight='bold')
        plt.title('Multiple Models - Confidence Score Distribution (Violin Plot)',
                 fontsize=25, fontweight='bold', pad=20)

        # Rotate X-axis labels 45 degrees
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=12)

        # Add background color
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')

        # Add grid lines
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add overall mean reference line
        all_scores = np.concatenate(violin_data)
        overall_mean = np.mean(all_scores)
        plt.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Overall Mean: {overall_mean:.2f}')

        # Add median points
        for i, model_name in enumerate(plot_df['display_model'].unique()):
            model_scores = plot_df[plot_df['display_model'] == model_name]['confidence_score']
            median_val = model_scores.median()

            # Add white circle at median position
            ax.plot(i, median_val, 'o', color='white', markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=10)

            # Optional: add value label on median point
            # ax.text(i, median_val, f'{median_val:.1f}', ha='center', va='bottom',
            #        fontsize=9, fontweight='bold', color='black')

        # Add legend
        plt.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)

        # Adjust layout
        plt.tight_layout()

        # Save violin plot
        output_file = os.path.join(self.output_dir, 'confidence_distribution_violin_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Multi-model confidence distribution violin plot saved: {output_file}")

        # Print statistics
        print("\nConfidence score statistics for each model (violin plot):")
        for i, model_name in enumerate(violin_labels):
            scores = violin_data[i]
            print(f"\nModel {model_name}:")
            print(f"  Mean: {scores.mean():.2f}")
            print(f"  Median: {scores.median():.2f}")
            print(f"  Std Dev: {scores.std():.2f}")
            print(f"  Min: {scores.min():.2f}")
            print(f"  Max: {scores.max():.2f}")
            print(f"  Sample count: {len(scores)}")
            # Calculate quartiles
            q25, q75 = np.percentile(scores, [25, 75])
            print(f"  25th percentile: {q25:.2f}")
            print(f"  75th percentile: {q75:.2f}")

        return {model: data for model, data in zip(violin_labels, violin_data)}

def main():
    parser = argparse.ArgumentParser(description='Plot confidence distribution charts')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory path containing CSV files')
    parser.add_argument('--output_dir', type=str, default='confidence_plots',
                       help='Output directory path for images')
    parser.add_argument('--files', nargs='+',
                       help='List of CSV files to process')

    args = parser.parse_args()

    # Initialize plotter
    plotter = ConfidenceDistributionPlotter(
        data_folder=args.data_dir,
        output_dir=args.output_dir
    )

    # Load data
    plotter.load_data(csv_files=args.files)

    # Plot confidence distribution for each model
    for model in plotter.models:
        plotter.plot_single_confidence_distribution(model)

    # Plot confidence distribution for multiple models
    if len(plotter.models) > 1:
        # Plot line chart
        plotter.plot_multiple_confidence_distributions()

        # Plot grouped bar chart
        plotter.plot_grouped_bar_chart()

        # Plot box plot
        plotter.plot_box_plot()

        # Plot violin plot
        plotter.plot_violin_plot()
    else:
        print("ℹ️ Only one model, skipping multi-model comparison charts")

if __name__ == "__main__":
    main()
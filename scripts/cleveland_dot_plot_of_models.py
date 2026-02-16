import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import re
import argparse
import os
import sys
from matplotlib.lines import Line2D

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze model performance data and generate visualizations')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file (required)')
    parser.add_argument('--output-dir', '-o', type=str, default='./',
                       help='Output directory for images (optional, default is current directory)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output image resolution (optional, default is 300)')
    parser.add_argument('--sort-by', type=str, default='AUC', choices=['AUC', 'AP', 'F1_Score'],
                       help='Metric to sort models by (optional, default is AUC)')
    return parser.parse_args()

def check_file_exists(file_path):
    """Check if file exists"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist!")
        sys.exit(1)

def clean_value(value):
    """Clean numerical data"""
    if pd.isna(value) or value == '':
        return 0.0
    # Remove special characters and extra dots
    cleaned = re.sub(r'\*\*|\.$|\(|\)|\s', '', str(value).strip())
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def clean_model_name(name):
    """Clean model names"""
    return re.sub(r'\*\*', '', str(name).strip())

def parse_cost(cost_str):
    """Parse price data, return average of input/output prices"""
    if pd.isna(cost_str) or cost_str == '':
        return None
    # Match number/number format
    match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)', str(cost_str))
    if match:
        input_cost = float(match.group(1))
        output_cost = float(match.group(2))
        return (input_cost + output_cost) / 2  # Return average price
    return None

def create_cleveland_dot_plot(df_clean, output_path, sort_by='AUC', dpi=300):
    """Create Cleveland Dot Plot showing all three metrics for each model (academic style)"""
    # Sort by specified metric (default AUC)
    df_sorted = df_clean.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # Set academic style
    sns.set_theme(style="white", font_scale=1.1)

    # Define academic color scheme and markers
    colors = {'AUC': '#004c6d', 'AP': '#c7522a', 'F1_Score': '#008f7a'}
    markers = {'AUC': 'o', 'AP': 's', 'F1_Score': '^'}  # Circle, square, triangle

    # Create figure - adjust size based on number of models
    n_models = len(df_sorted)
    fig_height = max(8, n_models * 0.3)  # 0.3 inches height per model
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.5, zorder=0)
    ax.grid(True, axis='x', linestyle=':', color='#ddd', alpha=0.5, zorder=0)

    metrics = ['AUC', 'AP', 'F1_Score']

    # Plot Cleveland dot chart
    for metric in metrics:
        ax.plot(
            df_sorted[metric],        # X-axis data
            df_sorted.index,          # Y-axis data (use index for positioning, replace labels later)
            marker=markers[metric],
            markersize=10,
            linestyle='None',         # Core: no lines, only points
            color=colors[metric],
            markeredgecolor='white',  # White edge makes points clearer
            markeredgewidth=1.0,
            alpha=0.9,
            label=metric,
            zorder=3
        )

    # Set Y-axis labels to model names
    ax.set_yticks(df_sorted.index)

    # Truncate long model names while maintaining readability
    y_labels = []
    for model in df_sorted['Model']:
        if len(model) > 40:
            y_labels.append(model[:18] + '...' + model[-19:])
        elif len(model) > 25:
            y_labels.append(model[:12] + '...' + model[-10:])
        else:
            y_labels.append(model)

    ax.set_yticklabels(y_labels, fontsize=10)

    # Title and axis labels - title font size increased to 25
    title = f"Model Performance Comparison: AUC, AP, and F1 Score\n(Models sorted by {sort_by}, descending)"
    ax.set_title(title, fontsize=25, fontweight='bold', pad=20, loc='left')
    ax.set_xlabel("Score Value (0.0 - 1.0)", fontsize=14, labelpad=10, fontweight='bold')
    ax.set_ylabel("")  # Remove Y-axis label, model names are sufficient

    # X-axis tick range
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Optimize legend - place in upper right corner inside chart
    legend = ax.legend(
        title="Metrics",
        title_fontsize='13',
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        edgecolor='#ccc',
        fancybox=False,  # Square border more academic
        fontsize=12
    )

    # Border control
    sns.despine(left=True, bottom=False, trim=True)  # Remove left and top borders
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('#333')

    # Hide left Y-axis tick marks, keep only labels
    ax.tick_params(axis='y', which='both', length=0)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Cleveland dot plot saved as: {output_path}")

def create_parallel_coordinates_plot(df_clean, output_path, dpi=300):
    """Create parallel coordinates plot"""
    # Find best models for each metric
    best_auc_model = df_clean.loc[df_clean['AUC'].idxmax()]
    best_ap_model = df_clean.loc[df_clean['AP'].idxmax()]
    best_f1_model = df_clean.loc[df_clean['F1_Score'].idxmax()]
    
    # Normalize data for parallel coordinates
    df_normalized = df_clean.copy()
    for col in ['AUC', 'AP', 'F1_Score']:
        df_normalized[col] = (df_clean[col] - df_clean[col].min()) / (df_clean[col].max() - df_clean[col].min())

    plt.figure(figsize=(14, 8))
    categories = ['AUC', 'AP', 'F1_Score']
    x_pos = range(len(categories))

    # Set special styles for best models
    best_models = [best_auc_model['Model'], best_ap_model['Model'], best_f1_model['Model']]

    # Draw lines for each model
    plotted_best_models = set()
    for i, row in df_clean.iterrows():
        values = [df_normalized.loc[i, 'AUC'], df_normalized.loc[i, 'AP'], df_normalized.loc[i, 'F1_Score']]
        
        # Check if it's a best model
        is_best = row['Model'] in best_models

        if is_best:
            color_map = {
                best_auc_model['Model']: '#004c6d',  # Use consistent blue from dot plot
                best_ap_model['Model']: '#c7522a',   # Use consistent orange from dot plot
                best_f1_model['Model']: '#008f7a'    # Use consistent green from dot plot
            }
            color = color_map[row['Model']]
            linewidth = 4
            alpha = 1.0
            if row['Model'] not in plotted_best_models:
                label = f"{row['Model']} ({'AUC' if row['Model'] == best_auc_model['Model'] else 'AP' if row['Model'] == best_ap_model['Model'] else 'F1'} Best)"
                plotted_best_models.add(row['Model'])
            else:
                label = None
        else:
            color = 'gray'
            linewidth = 1
            alpha = 0.3
            label = None
        
        plt.plot(x_pos, values, marker='o', linewidth=linewidth, alpha=alpha, 
                 color=color, label=label, markersize=6 if is_best else 3)

    plt.xticks(x_pos, categories, fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Score', fontsize=14, fontweight='bold')
    # Title font size increased to 25
    plt.title('Parallel Coordinates Plot for Multiple Metrics\n(Highlighting best performing models for each metric)',
              fontsize=25, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Parallel coordinates plot saved as: {output_path}")

def create_grouped_bar_chart(df_clean, output_path, dpi=300):
    """Create grouped bar chart for top models"""
    # Calculate composite score
    df_clean['Composite_Score'] = (df_clean['AUC'] + df_clean['AP'] + df_clean['F1_Score']) / 3
    top_12_models = df_clean.nlargest(12, 'Composite_Score')

    plt.figure(figsize=(16, 8))
    x = np.arange(len(top_12_models))
    width = 0.25

    # Use consistent color scheme from dot plot
    colors = {'AUC': '#004c6d', 'AP': '#c7522a', 'F1_Score': '#008f7a'}

    # Draw grouped bars
    bars1 = plt.bar(x - width, top_12_models['AUC'], width, label='AUC', 
                    alpha=0.8, color=colors['AUC'], edgecolor='darkblue', linewidth=1)
    bars2 = plt.bar(x, top_12_models['AP'], width, label='AP', 
                    alpha=0.8, color=colors['AP'], edgecolor='darkorange', linewidth=1)
    bars3 = plt.bar(x + width, top_12_models['F1_Score'], width, label='F1 Score', 
                    alpha=0.8, color=colors['F1_Score'], edgecolor='darkgreen', linewidth=1)

    # Add value labels
    def add_value_labels(bars, rotation=45):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=9, rotation=rotation, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    # Title font size increased to 25
    plt.title('Multi-Metric Comparison of Top 12 Models by Composite Score',
              fontsize=25, fontweight='bold')
    plt.xticks(x, [model[:12]+'...' if len(model) > 12 else model for model in top_12_models['Model']],
               rotation=45, ha='right', fontsize=11)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Grouped bar chart saved as: {output_path}")

def create_cost_performance_plot(df_clean, output_path, dpi=300):
    """Create cost-performance scatter plot"""
    # Filter models with price data
    df_with_cost = df_clean[df_clean['Avg_Cost'].notna()].copy()

    plt.figure(figsize=(12, 8))
    
    if len(df_with_cost) > 0:
        # Calculate performance-cost ratio (composite score / average cost)
        df_with_cost['Composite_Score'] = (df_with_cost['AUC'] + df_with_cost['AP'] + df_with_cost['F1_Score']) / 3
        df_with_cost['Performance_Cost_Ratio'] = df_with_cost['Composite_Score'] / df_with_cost['Avg_Cost']
        
        plt.scatter(df_with_cost['Avg_Cost'], df_with_cost['Composite_Score'], 
                    s=80, alpha=0.7, c=df_with_cost['Performance_Cost_Ratio'], cmap='viridis')
        
        # Add model labels (only for top performers to avoid clutter)
        for i, row in df_with_cost.nlargest(8, 'Composite_Score').iterrows():
            plt.annotate(row['Model'][:10] + '...',
                        (row['Avg_Cost'], row['Composite_Score']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        plt.xlabel('Average Cost (USD per million tokens)', fontsize=14, fontweight='bold')
        plt.ylabel('Composite Performance Score', fontsize=14, fontweight='bold')
        # Title font size increased to 25
        plt.title('Cost-Performance Analysis\n(Yellow color indicates better value)',
                  fontsize=25, fontweight='bold')
        plt.colorbar(label='Performance-Cost Ratio')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                 transform=plt.gca().transAxes, fontsize=18)
        # Title font size increased to 25
        plt.title('Cost-Performance Analysis (No Cost Data)', fontsize=25, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Cost-performance plot saved as: {output_path}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if input file exists
    check_file_exists(args.input)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    try:
        # Read CSV file
        df = pd.read_csv(args.input, encoding='utf-8-sig')
        print(f"Successfully read file: {args.input}")
        print(f"Data shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Please check file format and encoding (recommended: UTF-8 encoded CSV)")
        sys.exit(1)

    # Check for required columns
    required_columns = ['Model-v10', 'AUC', 'AP', 'F1 Score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: CSV file is missing required columns: {missing_columns}")
        print(f"Columns in CSV file: {list(df.columns)}")
        sys.exit(1)

    # Apply data cleaning
    df['Model'] = df['Model-v10'].apply(clean_model_name)
    df['AUC_cleaned'] = df['AUC'].apply(clean_value)
    df['AP_cleaned'] = df['AP'].apply(clean_value)
    df['F1_cleaned'] = df['F1 Score'].apply(clean_value)
    
    # Get price column name (might be in Chinese or English)
    cost_column = None
    for col in df.columns:
        if 'price' in col.lower() or 'cost' in col.lower():
            cost_column = col
            break
    
    if cost_column:
        df['Avg_Cost'] = df[cost_column].apply(parse_cost)
        print(f"Found cost column: {cost_column}")
    else:
        df['Avg_Cost'] = None
        print("No cost column found, will skip cost-performance analysis")

    # Create cleaned data frame
    df_clean = pd.DataFrame({
        'Model': df['Model'],
        'AUC': df['AUC_cleaned'],
        'AP': df['AP_cleaned'],
        'F1_Score': df['F1_cleaned'],
        'Avg_Cost': df['Avg_Cost']
    })

    # Find best models for each metric
    best_auc_model = df_clean.loc[df_clean['AUC'].idxmax()]
    best_ap_model = df_clean.loc[df_clean['AP'].idxmax()]
    best_f1_model = df_clean.loc[df_clean['F1_Score'].idxmax()]

    print("="*70)
    print("Best Models for Each Metric")
    print("="*70)
    print(f"AUC  Best: {best_auc_model['Model']:30} - {best_auc_model['AUC']:.3f}")
    print(f"AP   Best: {best_ap_model['Model']:30} - {best_ap_model['AP']:.3f}")
    print(f"F1   Best: {best_f1_model['Model']:30} - {best_f1_model['F1_Score']:.3f}")
    print("="*70)

    # Generate all four plots
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # 1. Cleveland Dot Plot (replaces original line chart)
    dot_output = os.path.join(args.output_dir, f"{base_name}_cleveland_dot_plot.png")
    create_cleveland_dot_plot(df_clean, dot_output, args.sort_by, args.dpi)
    
    # 2. Parallel Coordinates Plot
    parallel_output = os.path.join(args.output_dir, f"{base_name}_parallel_coordinates.png")
    create_parallel_coordinates_plot(df_clean, parallel_output, args.dpi)
    
    # 3. Grouped Bar Chart
    bar_output = os.path.join(args.output_dir, f"{base_name}_grouped_bar.png")
    create_grouped_bar_chart(df_clean, bar_output, args.dpi)
    
    # 4. Cost-Performance Plot
    cost_output = os.path.join(args.output_dir, f"{base_name}_cost_performance.png")
    create_cost_performance_plot(df_clean, cost_output, args.dpi)

    # Output detailed analysis report
    print("\n" + "="*70)
    print("Detailed Performance Analysis Report")
    print("="*70)

    # Top 5 models for each metric
    for metric in ['AUC', 'AP', 'F1_Score']:
        print(f"\n{metric} Top 5:")
        top_5 = df_clean.nlargest(5, metric)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            cost_info = f" | Avg Cost: ${row['Avg_Cost']:.2f}" if pd.notna(row['Avg_Cost']) else " | Cost: Unknown"
            print(f"  {i}. {row['Model']:30} {metric}: {row[metric]:.3f}{cost_info}")

    # Best overall models
    df_clean['Composite_Score'] = (df_clean['AUC'] + df_clean['AP'] + df_clean['F1_Score']) / 3
    print(f"\nBest Overall Models (Average Score):")
    top_10_composite = df_clean.nlargest(10, 'Composite_Score')
    for i, (_, row) in enumerate(top_10_composite.iterrows(), 1):
        cost_info = f" | Avg Cost: ${row['Avg_Cost']:.2f}" if pd.notna(row['Avg_Cost']) else " | Cost: Unknown"
        print(f"  {i}. {row['Model']:30} Composite: {row['Composite_Score']:.3f} (AUC: {row['AUC']:.3f}, AP: {row['AP']:.3f}, F1: {row['F1_Score']:.3f}){cost_info}")

    # Statistics for each metric
    print(f"\nMetric Statistics:")
    for metric in ['AUC', 'AP', 'F1_Score']:
        print(f"  {metric}: Min={df_clean[metric].min():.3f}, Max={df_clean[metric].max():.3f}, Mean={df_clean[metric].mean():.3f}, Std={df_clean[metric].std():.3f}")

    # Cost-performance analysis (if cost data available)
    df_with_cost = df_clean[df_clean['Avg_Cost'].notna()].copy()
    if len(df_with_cost) > 0:
        df_with_cost['Performance_Cost_Ratio'] = df_with_cost['Composite_Score'] / df_with_cost['Avg_Cost']
        print(f"\nCost-Performance Analysis (based on {len(df_with_cost)} models with cost data):")
        top_5_value = df_with_cost.nlargest(5, 'Performance_Cost_Ratio')
        for i, (_, row) in enumerate(top_5_value.iterrows(), 1):
            print(f"  {i}. {row['Model']:30} Value: {row['Performance_Cost_Ratio']:.2f} (Composite: {row['Composite_Score']:.3f}, Avg Cost: ${row['Avg_Cost']:.2f})")

    print(f"\nAll charts saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
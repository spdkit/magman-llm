#!/usr/bin/env python3
"""
LLM Evaluation Results Analysis Script
Supports ROC curves, AUC calculation, and various performance metrics analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import argparse
from typing import Dict, List, Tuple, Optional
import json

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from roc_utils import get_roc_probabilities, validate_confidence_scores

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultsAnalyzer:
    """Evaluation results analyzer"""

    def __init__(self, results_file: str, output_dir: str = None, debug: bool = False):
        """
        Initialize analyzer

        Args:
            results_file: Results file path
            output_dir: Output directory, auto-generated based on input filename if None
            debug: Whether to enable debug mode
        """
        self.results_file = results_file
        self.debug = debug
        self.df = None
        self.models = []

        # Auto-generate output directory name
        if output_dir is None:
            base_name = os.path.splitext(os.path.basename(results_file))[0]
            self.output_dir = f"{base_name}_analysis"
        else:
            self.output_dir = output_dir

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess results data"""
        try:
            self.df = pd.read_csv(self.results_file)
            print(f"✅ Successfully loaded results data: {len(self.df)} records")

            # Check data columns
            required_columns = ['prediction', 'true_label', 'success']
            missing_columns = [col for col in required_columns if col not in self.df.columns]

            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return False

            # Filter successful results
            self.df = self.df[self.df['success'] == True].copy()
            print(f"✅ Successful evaluation results: {len(self.df)} records")

            if len(self.df) == 0:
                print("❌ No successful evaluation results")
                return False

            # Ensure prediction values are numeric
            self.df['prediction'] = pd.to_numeric(self.df['prediction'], errors='coerce')
            self.df = self.df.dropna(subset=['prediction'])

            # Check if model information exists
            if 'model' in self.df.columns:
                self.models = self.df['model'].unique().tolist()
                print(f"📊 Found results for {len(self.models)} models")
            else:
                # Single model analysis, no model column needed
                self.models = ['single_model']
                print("📊 Analyzing single model results")

            return True
            return True

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Calculate various performance metrics

        Args:
            predictions: Prediction values
            true_labels: True labels

        Returns:
            Performance metrics dictionary
        """
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        # Specificity and other metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2

        # AUC (based on confidence scores)
        auc_score = None
        probabilities = self._get_confidence_scores(self.df)
        if probabilities is not None and len(probabilities) > 0:
            try:
                auc_score = roc_auc_score(true_labels, probabilities)
            except Exception:
                pass
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'auc': auc_score,
            'total_samples': len(true_labels)
        }
    
    def plot_roc_curve(self, model_name: str = None):
        """
        Plot ROC curve

        Args:
            model_name: Model name, uses first model if None
        """
        if model_name is None:
            model_name = self.models[0]

        # Filter data for specific model
        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        true_labels = model_df['true_label'].values

        # Get confidence scores (supports multiple column names)
        probabilities = self._get_confidence_scores(model_df)

        if probabilities is None:
            print(f"⚠️ Model {model_name} has no confidence values, cannot plot ROC curve")
            return
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        if len(self.models) > 1:
            plt.title(f'{model_name} - ROC Curve', fontsize=14)
        else:
            plt.title('ROC Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save figure
        if len(self.models) > 1:
            output_file = os.path.join(self.output_dir, f'roc_curve_{model_name}.png')
        else:
            output_file = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ ROC curve saved: {output_file}")
        
        return roc_auc
    
    def _get_confidence_scores(self, model_df):
        """
        Uniformly get confidence scores using standard ROC probability mapping

        Args:
            model_df: Model data DataFrame

        Returns:
            Normalized probability array, or None
        """
        # Check if valid confidence scores exist
        if validate_confidence_scores(model_df, 'confidence_score'):
            try:
                # Use unified ROC probability calculation
                roc_probs = get_roc_probabilities(model_df, 'confidence_score', 'prediction')
                return roc_probs
            except Exception as e:
                if self.debug:
                    print(f"ROC probability calculation failed: {e}")
                return None

        # If no confidence scores, try other columns (backward compatibility)
        confidence_cols = ['confidence', 'probability', 'roc_probability']

        for col in confidence_cols:
            if col in model_df.columns:
                values = pd.to_numeric(model_df[col], errors='coerce')
                if values.notna().all() and len(values) > 0:
                    # Normalize to 0-1 range
                    max_val = values.max()
                    if max_val <= 1.0:
                        return values.values
                    elif max_val <= 100.0:
                        return values.values / 100.0
                    else:
                        return values.values / 10.0

        return None

    def plot_confusion_matrix(self, model_name: str = None):
        """
        Plot confusion matrix

        Args:
            model_name: Model name
        """
        if model_name is None:
            model_name = self.models[0]

        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        true_labels = model_df['true_label'].values
        predictions = model_df['prediction'].values

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-convergent', 'Convergent'],
                   yticklabels=['Non-convergent', 'Convergent'])
        if len(self.models) > 1:
            plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
        else:
            plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # Save figure
        if len(self.models) > 1:
            output_file = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
        else:
            output_file = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Confusion matrix saved: {output_file}")
    
    def plot_precision_recall_curve(self, model_name: str = None):
        """
        Plot precision-recall curve

        Args:
            model_name: Model name
        """
        if model_name is None:
            model_name = self.models[0]

        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        # Get confidence scores
        probabilities = self._get_confidence_scores(model_df)

        if probabilities is None:
            print(f"⚠️ Model {model_name} has no confidence values, cannot plot PR curve")
            return

        true_labels = model_df['true_label'].values

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
        avg_precision = average_precision_score(true_labels, probabilities)

        # Plot PR curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')

        # Add baseline (random classifier performance)
        positive_ratio = np.mean(true_labels)
        plt.axhline(y=positive_ratio, color='red', linestyle='--',
                   label=f'Random classifier (AP = {positive_ratio:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Recall)', fontsize=12)
        plt.ylabel('Precision (Precision)', fontsize=12)
        if len(self.models) > 1:
            plt.title(f'{model_name} - Precision-Recall Curve', fontsize=14)
        else:
            plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        # Save figure
        if len(self.models) > 1:
            output_file = os.path.join(self.output_dir, f'pr_curve_{model_name}.png')
        else:
            output_file = os.path.join(self.output_dir, 'pr_curve.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ PR curve saved: {output_file}")

        return avg_precision
    
    def plot_confidence_distribution(self, model_name: str = None):
        """
        Plot confidence score distribution

        Args:
            model_name: Model name
        """
        if model_name is None:
            model_name = self.models[0]

        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        if 'confidence_score' not in model_df.columns:
            print(f"⚠️ Model {model_name} has no confidence score data")
            return

        # Count confidence score distribution
        confidence_scores = model_df['confidence_score'].value_counts().sort_index()

        # Plot confidence distribution
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(confidence_scores)))
        bars = plt.bar(confidence_scores.index, confidence_scores.values, color=colors)

        plt.xlabel('Confidence Score (0-9)', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        if len(self.models) > 1:
            plt.title(f'{model_name} - Confidence Score Distribution', fontsize=14)
        else:
            plt.title('Confidence Score Distribution', fontsize=14)
        plt.xticks(range(0, 10))
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        # Save figure
        if len(self.models) > 1:
            output_file = os.path.join(self.output_dir, f'confidence_distribution_{model_name}.png')
        else:
            output_file = os.path.join(self.output_dir, 'confidence_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Confidence distribution saved: {output_file}")

        # Print distribution statistics
        print("Confidence score distribution:")
        for score, count in confidence_scores.items():
            print(f"   {score}: {count} samples")

        return confidence_scores.to_dict()
    
    def generate_plotting_code(self, model_name: str = None):
        """
        Generate pure plotting code and data files
        Save processed data and independent plotting script
        """
        if model_name is None:
            model_name = self.models[0]

        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        # Prepare plotting data
        true_labels = model_df['true_label'].values
        predictions = model_df['prediction'].values

        # Get confidence scores for ROC calculation
        roc_probs = []
        confidence_scores = []

        if 'confidence_score' in model_df.columns and 'prediction' in model_df.columns:
            for _, row in model_df.iterrows():
                if pd.isna(row['confidence_score']) or pd.isna(row['prediction']):
                    roc_probs.append(np.nan)
                    confidence_scores.append(np.nan)
                else:
                    # Convert 0-9 confidence score to 0.05-0.95 probability
                    base_score = 0.05 + (row['confidence_score'] / 9.0) * 0.9
                    # Adjust score based on prediction direction
                    if row['prediction'] == 1:  # Convergence prediction
                        roc_probs.append(base_score)
                    else:  # Non-convergence prediction
                        roc_probs.append(1.0 - base_score)
                    confidence_scores.append(int(row['confidence_score']))
            roc_probs = np.array(roc_probs)
            confidence_scores = np.array(confidence_scores)
        else:
            roc_probs = None
            confidence_scores = None

        # Save processed data
        processed_data = {
            'true_labels': true_labels.tolist(),
            'predictions': predictions.tolist(),
            'roc_probabilities': roc_probs.tolist() if roc_probs is not None else [],
            'confidence_scores': confidence_scores.tolist() if confidence_scores is not None else [],
            'model_name': model_name,
            'total_samples': len(true_labels),
            'convergent_samples': int(sum(true_labels)),
            'non_convergent_samples': int(len(true_labels) - sum(true_labels))
        }

        data_file = os.path.join(self.output_dir, 'processed_data.json')
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Processed data saved: {data_file}")
        
        # Generate pure plotting code
        plotting_code = '''#!/usr/bin/env python3
"""
Pure plotting code - Based on processed data file
Data source: processed_data.json
Ensure before running this script processed_data.json file exists
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load processed data"""
    with open('processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert tonumpyarray
    true_labels = np.array(data['true_labels'])
    predictions = np.array(data['predictions'])
    roc_probabilities = np.array(data['roc_probabilities']) if data['roc_probabilities'] else None
    confidence_scores = np.array(data['confidence_scores']) if data['confidence_scores'] else None
    
    return {
        'true_labels': true_labels,
        'predictions': predictions,
        'roc_probabilities': roc_probabilities,
        'confidence_scores': confidence_scores,
        'model_name': data['model_name'],
        'total_samples': data['total_samples'],
        'convergent_samples': data['convergent_samples'],
        'non_convergent_samples': data['non_convergent_samples']
    }

def plot_roc_curve(data):
    """PlotROCcurve"""
    true_labels = data['true_labels']
    roc_probabilities = data['roc_probabilities']
    
    if roc_probabilities is None or len(roc_probabilities) == 0:
        print("⚠️ NoROCprobability data, skippingROCcurve")
        return None
    
    # RemoveNaNvalue
    mask = ~np.isnan(roc_probabilities)
    true_labels_clean = true_labels[mask]
    roc_probabilities_clean = roc_probabilities[mask]
    
    if len(true_labels_clean) == 0:
        print("⚠️ Valid data is empty, skippingROCcurve")
        return None
    
    fpr, tpr, thresholds = roc_curve(true_labels_clean, roc_probabilities_clean)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROCcurve (AUC = %.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROCcurve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ ROCcurvesaved: roc_curve.png (AUC = %.3f)" % roc_auc)
    return roc_auc

def plot_confusion_matrix(data):
    """Plotconfusion matrix"""
    true_labels = data['true_labels']
    predictions = data['predictions']
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-convergent', 'Convergent'],
               yticklabels=['Non-convergent', 'Convergent'])
    plt.title('confusion matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ confusion matrixsaved: confusion_matrix.png")
    return cm

def plot_pr_curve(data):
    """PlotPrecision-Recallcurve"""
    true_labels = data['true_labels']
    roc_probabilities = data['roc_probabilities']
    
    if roc_probabilities is None or len(roc_probabilities) == 0:
        print("⚠️ NoROCprobability data, skippingPRcurve")
        return None
    
    # RemoveNaNvalue
    mask = ~np.isnan(roc_probabilities)
    true_labels_clean = true_labels[mask]
    roc_probabilities_clean = roc_probabilities[mask]
    
    if len(true_labels_clean) == 0:
        print("⚠️ Valid data is empty, skippingPRcurve")
        return None
    
    precision, recall, thresholds = precision_recall_curve(true_labels_clean, roc_probabilities_clean)
    avg_precision = average_precision_score(true_labels_clean, roc_probabilities_clean)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='PRcurve (AP = %.3f)' % avg_precision)
    
    # AddBaseline
    positive_ratio = np.mean(true_labels_clean)
    plt.axhline(y=positive_ratio, color='red', linestyle='--', 
               label='Random classifier (AP = %.3f)' % positive_ratio)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Recall)', fontsize=12)
    plt.ylabel('Precision (Precision)', fontsize=12)
    plt.title('Precision-Recallcurve', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ PRcurvesaved: pr_curve.png (AP = %.3f)" % avg_precision)
    return avg_precision

def plot_confidence_distribution(data):
    """PlotConfidence scoredistribution"""
    confidence_scores = data['confidence_scores']
    
    if confidence_scores is None or len(confidence_scores) == 0:
        print("⚠️ NoConfidence scoredata，Skipdistribution plot")
        return {}
    
    # Statisticsdistribution
    valid_scores = confidence_scores[~np.isnan(confidence_scores)]
    if len(valid_scores) == 0:
        print("⚠️ NovalidConfidence score")
        return {}
    
    unique_scores, counts = np.unique(valid_scores, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_scores)))
    bars = plt.bar(unique_scores, counts, color=colors, tick_label=unique_scores)
    
    plt.xlabel('Confidence score (0-9)', fontsize=12)
    plt.ylabel('Number of samples', fontsize=12)
    plt.title('Confidence scoredistribution', fontsize=14)
    plt.xticks(range(0, 10))
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                '%d' % int(height), ha='center', va='bottom')
    
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PrintdistributionStatistics
    distribution = dict(zip(unique_scores.astype(int), counts))
    print("Confidence scoredistribution:")
    for score in sorted(distribution.keys()):
        print("   %d: %d samples" % (score, distribution[score]))
    
    print("✅ confidence distributionchartsaved: confidence_distribution.png")
    return distribution

def main():
    """Main plotting function"""
    print("=== Pure plotting code ===")
    print("LoadingLoad processed data...")
    
    try:
        data = load_data()
    except FileNotFoundError:
        print("❌ Cannot find processed_data.json file, ensure file is in current directory")
        return
    except Exception as e:
        print("❌ Load dataFailed: %s" % str(e))
        return
    
    # DisplayData information
    print("")
    print("Data information:")
    print("Model name: %s" % data['model_name'])
    print("Total samples: %d" % data['total_samples'])
    print("Convergent samples: %d" % data['convergent_samples'])
    print("notConvergent samples: %d" % data['non_convergent_samples'])
    
    # Plotchart
    print("")
    print("=" * 50)
    print("1. PlotROCcurve...")
    auc_score = plot_roc_curve(data)
    
    print("")
    print("=" * 50)
    print("2. Plotconfusion matrix...")
    cm = plot_confusion_matrix(data)
    
    print("")
    print("=" * 50)
    print("3. PlotPRcurve...")
    ap_score = plot_pr_curve(data)
    
    print("")
    print("=" * 50)
    print("4. Plotconfidence distribution...")
    confidence_dist = plot_confidence_distribution(data)
    
    print("")
    print("=" * 50)
    print("🎉 AllchartPlotcomplete！")
    print("Generated files:")
    print("  - roc_curve.png")
    print("  - confusion_matrix.png")
    print("  - pr_curve.png")
    print("  - confidence_distribution.png")

if __name__ == "__main__":
    main()
'''
        
        # SavePure plotting code
        code_file = os.path.join(self.output_dir, 'plotting_code.py')
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(plotting_code)
        
        print(f"✅ Pure plotting codesaved: {code_file}")
        
        return code_file
    
    def analyze_model(self, model_name: str = None) -> Dict:
        """
        Analyze single model performance

        Args:
            model_name: Model name

        Returns:
            Analysis results dictionary
        """
        if model_name is None:
            model_name = self.models[0]

        if 'model' in self.df.columns:
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df

        if len(self.models) > 1:
            print(f"\n📊 Analyzing model: {model_name}")
        else:
            print(f"\n📊 Analysis results")
        print(f"Sample count: {len(model_df)}")

        # Calculate basic metrics
        predictions = model_df['prediction'].values
        true_labels = model_df['true_label'].values

        metrics = self.calculate_metrics(predictions, true_labels)

        # Print basic metrics
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"Specificity: {metrics['specificity']:.3f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")

        if metrics['auc'] is not None:
            print(f"AUC: {metrics['auc']:.3f}")

        # Plot charts
        self.plot_roc_curve(model_name)
        self.plot_confusion_matrix(model_name)
        self.plot_precision_recall_curve(model_name)
        self.plot_confidence_distribution(model_name)

        return metrics
    
    
    
    def generate_report(self):
        """
        Generate complete analysis report
        """
        print(f"\n📊 Generating complete analysis report")
        print("=" * 80)

        report = {
            'data_info': {
                'total_samples': len(self.df),
                'models': self.models,
                'success_rate': len(self.df) / len(pd.read_csv(self.results_file))
            },
            'model_results': {}
        }

        # Analyze each model
        for model in self.models:
            print(f"\nAnalyzing model: {model}")
            metrics = self.analyze_model(model)
            report['model_results'][model] = metrics


        # Save report
        report_file = os.path.join(self.output_dir, 'analysis_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✅ Complete analysis report saved: {report_file}")

        # Generate plotting code
        self.generate_plotting_code()

        return report


def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Results Analysis')
    parser.add_argument('results_file', help='Evaluation results file path')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory, auto-generated based on input filename if None')
    parser.add_argument('--model', '-m', help='Specify model name to analyze')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.results_file):
        print(f"❌ Results file does not exist: {args.results_file}")
        return

    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_file, args.output_dir, args.debug)

    if not analyzer.df is not None and len(analyzer.df) > 0:
        return

    # Execute analysis based on parameters
    if args.model:
        analyzer.analyze_model(args.model)
    else:
        analyzer.generate_report()


if __name__ == "__main__":
    main()


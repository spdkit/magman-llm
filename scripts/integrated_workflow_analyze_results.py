#!/usr/bin/env python3
"""
LLM Evaluation Results Analysis Script
Supports ROC curves, AUC calculation and multiple performance metrics analysis
Enhanced version: Supports VASP dynamic step prediction analysis and path handling
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
try:
    from roc_utils import get_roc_probabilities, validate_confidence_scores
except ImportError:
    # Fallback functions if import fails
    def validate_confidence_scores(df, column):
        return column in df.columns and not df[column].isna().all()
    
    def get_roc_probabilities(df, confidence_col, prediction_col):
        probabilities = []
        for _, row in df.iterrows():
            if pd.isna(row[confidence_col]) or pd.isna(row[prediction_col]):
                probabilities.append(np.nan)
            else:
                base_score = 0.05 + (row[confidence_col] / 9.0) * 0.9
                if row[prediction_col] == 1:
                    probabilities.append(base_score)
                else:
                    probabilities.append(1.0 - base_score)
        return np.array(probabilities)

# Set Chinese fonts (keeping for compatibility, but using English)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class ResultsAnalyzer:
    """Evaluation Results Analyzer"""
    
    def __init__(self, results_file: str, output_dir: str = None, debug: bool = False):
        """
        Initialize analyzer
        
        Args:
            results_file: Path to results file
            output_dir: Output directory, if None automatically generated based on input filename
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
            # Enhanced file path handling
            file_path = self._resolve_file_path(self.results_file)
            if file_path is None:
                print(f"❌ Cannot find results file: {self.results_file}")
                return False
            
            # Try multiple encoding methods to read file
            encodings = ['utf-8-sig', 'utf-8', 'latin-1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ Successfully loaded data using {encoding}: {len(self.df)} records")
                    break
                except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
                    if self.debug:
                        print(f"Encoding {encoding} failed: {e}")
                    continue
            
            if self.df is None:
                print("❌ Unable to read file with any encoding")
                return False
            
            # Check data columns
            required_columns = ['true_label']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Handle prediction column names (support multiple naming conventions)
            prediction_columns = ['prediction', 'final_prediction']
            prediction_col = None
            for col in prediction_columns:
                if col in self.df.columns:
                    prediction_col = col
                    break
            
            if prediction_col is None:
                print("❌ Cannot find prediction results column")
                return False
            
            # Rename prediction column to standard name
            if prediction_col != 'prediction':
                self.df['prediction'] = self.df[prediction_col]
            
            # Handle confidence column names
            confidence_columns = ['confidence_score', 'final_confidence', 'confidence']
            confidence_col = None
            for col in confidence_columns:
                if col in self.df.columns:
                    confidence_col = col
                    break
            
            if confidence_col and confidence_col != 'confidence_score':
                self.df['confidence_score'] = self.df[confidence_col]
            
            # Handle status column (filter successful results)
            if 'success' in self.df.columns:
                # Use success column for filtering
                self.df = self.df[self.df['success'] == True].copy()
            elif 'status' in self.df.columns:
                # Use status column to determine success status
                valid_status = ['Continue', 'Kill']
                self.df = self.df[self.df['status'].isin(valid_status)].copy()
            else:
                # No status column, use all data
                print("⚠️ No status column, using all data")
            
            print(f"✅ Valid evaluation results: {len(self.df)} records")
            
            if len(self.df) == 0:
                print("❌ No valid evaluation results")
                return False
            
            # Ensure prediction values are numeric
            self.df['prediction'] = pd.to_numeric(self.df['prediction'], errors='coerce')
            self.df['true_label'] = pd.to_numeric(self.df['true_label'], errors='coerce')
            
            # Handle step columns (for step saving analysis)
            if 'total_steps' in self.df.columns and 'predicted_at_steps' in self.df.columns:
                self.df['total_steps'] = pd.to_numeric(self.df['total_steps'], errors='coerce')
                self.df['predicted_at_steps'] = pd.to_numeric(self.df['predicted_at_steps'], errors='coerce')
            
            self.df = self.df.dropna(subset=['prediction', 'true_label'])
            
            # Check for model information
            if 'model' in self.df.columns:
                self.models = self.df['model'].unique().tolist()
                print(f"📊 Found results for {len(self.models)} models")
            else:
                # Single model analysis, no model column needed
                self.models = ['single_model']
                print("📊 Analyzing single model results")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _resolve_file_path(self, file_path):
        """
        Resolve file path, try multiple possibilities
        
        Args:
            file_path: Original file path
            
        Returns:
            Resolved valid file path, or None
        """
        # List of paths to try
        possible_paths = [
            file_path,  # Original path
            os.path.abspath(file_path),  # Absolute path
            os.path.join(os.path.dirname(__file__), file_path),  # Relative to script
            os.path.join(os.path.dirname(__file__), '..', file_path),  # Relative to script parent directory
            os.path.join(os.getcwd(), file_path),  # Relative to current working directory
        ]
        
        # If path contains directory, also try just filename
        if os.path.dirname(file_path):
            possible_paths.append(os.path.basename(file_path))
        
        # Check each path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found file: {path}")
                return path
        
        # If not found, print debug information
        print("❌ File does not exist, tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        
        return None

    def calculate_step_saving(self):
        """
        Calculate step saving statistics
        
        Returns:
            Step saving statistics dictionary
        """
        # Check for required columns
        if 'total_steps' not in self.df.columns or 'predicted_at_steps' not in self.df.columns:
            print("⚠️ Missing step data, cannot calculate step savings")
            return None
        
        # Filter valid data
        valid_df = self.df[
            (self.df['total_steps'] > 0) & 
            (self.df['predicted_at_steps'] > 0) &
            (self.df['total_steps'] >= self.df['predicted_at_steps'])
        ].copy()
        
        if len(valid_df) == 0:
            print("❌ No valid step data")
            return None
        
        # Calculate saved steps and ratio
        valid_df['steps_saved'] = valid_df['total_steps'] - valid_df['predicted_at_steps']
        valid_df['saving_ratio'] = (valid_df['steps_saved'] / valid_df['total_steps']) * 100
        
        # Overall statistics
        total_original_steps = valid_df['total_steps'].sum()
        total_predicted_steps = valid_df['predicted_at_steps'].sum()
        total_steps_saved = total_original_steps - total_predicted_steps
        overall_saving_ratio = (total_steps_saved / total_original_steps) * 100
        
        # Group statistics by true label
        convergent_df = valid_df[valid_df['true_label'] == 1]
        non_convergent_df = valid_df[valid_df['true_label'] == 0]
        
        stats = {
            'overall': {
                'total_samples': len(valid_df),
                'total_original_steps': int(total_original_steps),
                'total_predicted_steps': int(total_predicted_steps),
                'total_steps_saved': int(total_steps_saved),
                'saving_ratio': round(overall_saving_ratio, 2),
                'avg_saving_ratio': round(valid_df['saving_ratio'].mean(), 2),
                'avg_steps_saved': round(valid_df['steps_saved'].mean(), 2)
            },
            'by_true_label': {
                'convergent': {
                    'samples': len(convergent_df),
                    'total_original_steps': int(convergent_df['total_steps'].sum()) if len(convergent_df) > 0 else 0,
                    'total_predicted_steps': int(convergent_df['predicted_at_steps'].sum()) if len(convergent_df) > 0 else 0,
                    'saving_ratio': round((convergent_df['saving_ratio'].mean() if len(convergent_df) > 0 else 0), 2),
                    'avg_steps_saved': round(convergent_df['steps_saved'].mean() if len(convergent_df) > 0 else 0, 2)
                },
                'non_convergent': {
                    'samples': len(non_convergent_df),
                    'total_original_steps': int(non_convergent_df['total_steps'].sum()) if len(non_convergent_df) > 0 else 0,
                    'total_predicted_steps': int(non_convergent_df['predicted_at_steps'].sum()) if len(non_convergent_df) > 0 else 0,
                    'saving_ratio': round((non_convergent_df['saving_ratio'].mean() if len(non_convergent_df) > 0 else 0), 2),
                    'avg_steps_saved': round(non_convergent_df['steps_saved'].mean() if len(non_convergent_df) > 0 else 0, 2)
                }
            }
        }
        
        return stats
    
    def plot_step_saving_analysis(self, model_name: str = None):
        """
        Plot step saving analysis
        
        Args:
            model_name: Model name
        """
        # First check if step data exists
        if 'total_steps' not in self.df.columns or 'predicted_at_steps' not in self.df.columns:
            print("⚠️ Missing step data, cannot perform step saving analysis")
            return None
        
        # Filter valid data
        valid_df = self.df[
            (self.df['total_steps'] > 0) & 
            (self.df['predicted_at_steps'] > 0) &
            (self.df['total_steps'] >= self.df['predicted_at_steps'])
        ].copy()
        
        if len(valid_df) == 0:
            print("❌ No valid step data")
            return None
        
        # Calculate saved steps and ratio
        valid_df['steps_saved'] = valid_df['total_steps'] - valid_df['predicted_at_steps']
        valid_df['saving_ratio'] = (valid_df['steps_saved'] / valid_df['total_steps']) * 100
        
        # Overall statistics
        total_original_steps = valid_df['total_steps'].sum()
        total_predicted_steps = valid_df['predicted_at_steps'].sum()
        total_steps_saved = total_original_steps - total_predicted_steps
        overall_saving_ratio = (total_steps_saved / total_original_steps) * 100
        
        # Group statistics by true label
        convergent_df = valid_df[valid_df['true_label'] == 1]
        non_convergent_df = valid_df[valid_df['true_label'] == 0]
        
        stats = {
            'overall': {
                'total_samples': len(valid_df),
                'total_original_steps': int(total_original_steps),
                'total_predicted_steps': int(total_predicted_steps),
                'total_steps_saved': int(total_steps_saved),
                'saving_ratio': round(overall_saving_ratio, 2),
                'avg_saving_ratio': round(valid_df['saving_ratio'].mean(), 2),
                'avg_steps_saved': round(valid_df['steps_saved'].mean(), 2)
            },
            'by_true_label': {
                'convergent': {
                    'samples': len(convergent_df),
                    'total_original_steps': int(convergent_df['total_steps'].sum()) if len(convergent_df) > 0 else 0,
                    'total_predicted_steps': int(convergent_df['predicted_at_steps'].sum()) if len(convergent_df) > 0 else 0,
                    'saving_ratio': round((convergent_df['saving_ratio'].mean() if len(convergent_df) > 0 else 0), 2),
                    'avg_steps_saved': round(convergent_df['steps_saved'].mean() if len(convergent_df) > 0 else 0, 2)
                },
                'non_convergent': {
                    'samples': len(non_convergent_df),
                    'total_original_steps': int(non_convergent_df['total_steps'].sum()) if len(non_convergent_df) > 0 else 0,
                    'total_predicted_steps': int(non_convergent_df['predicted_at_steps'].sum()) if len(non_convergent_df) > 0 else 0,
                    'saving_ratio': round((non_convergent_df['saving_ratio'].mean() if len(non_convergent_df) > 0 else 0), 2),
                    'avg_steps_saved': round(non_convergent_df['steps_saved'].mean() if len(non_convergent_df) > 0 else 0, 2)
                }
            }
        }
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Step Saving Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall step allocation pie chart
        sizes = [stats['overall']['total_predicted_steps'], stats['overall']['total_steps_saved']]
        labels = [f'Steps Used\n{sizes[0]:,}', f'Steps Saved\n{sizes[1]:,}']
        colors = ['#66b3ff', '#ff9999']
        axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title(f'Overall Step Allocation\n(Saving Ratio: {stats["overall"]["saving_ratio"]}%)')
        
        # 2. Convergent vs non-convergent saving ratio
        categories = ['Convergent Samples', 'Non-convergent Samples']
        saving_ratios = [
            stats['by_true_label']['convergent']['saving_ratio'],
            stats['by_true_label']['non_convergent']['saving_ratio']
        ]
        bars = axes[0,1].bar(categories, saving_ratios, color=['lightgreen', 'lightcoral'])
        axes[0,1].set_ylabel('Average Saving Ratio (%)')
        axes[0,1].set_title('Step Saving by Sample Type')
        for i, v in enumerate(saving_ratios):
            axes[0,1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        
        # 3. Saving ratio distribution histogram
        saving_ratios_data = valid_df['saving_ratio']
        axes[0,2].hist(saving_ratios_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,2].set_xlabel('Step Saving Ratio (%)')
        axes[0,2].set_ylabel('Number of Samples')
        axes[0,2].set_title('Step Saving Ratio Distribution')
        axes[0,2].axvline(saving_ratios_data.mean(), color='red', linestyle='--', 
                         label=f'Mean: {saving_ratios_data.mean():.1f}%')
        axes[0,2].legend()
        
        # 4. Original steps vs predicted steps scatter plot
        convergent_mask = valid_df['true_label'] == 1
        axes[1,0].scatter(valid_df[convergent_mask]['total_steps'], 
                         valid_df[convergent_mask]['predicted_at_steps'], 
                         alpha=0.6, color='green', label='Convergent Samples')
        axes[1,0].scatter(valid_df[~convergent_mask]['total_steps'], 
                         valid_df[~convergent_mask]['predicted_at_steps'], 
                         alpha=0.6, color='red', label='Non-convergent Samples')
        
        # Add diagonal line (y=x)
        max_steps = max(valid_df['total_steps'].max(), valid_df['predicted_at_steps'].max())
        axes[1,0].plot([0, max_steps], [0, max_steps], 'k--', alpha=0.5, label='y=x')
        axes[1,0].set_xlabel('Total Original Steps')
        axes[1,0].set_ylabel('Prediction Step')
        axes[1,0].set_title('Step Usage Scatter Plot')
        axes[1,0].legend()
        
        # 5. Saving ratio by prediction accuracy
        valid_df['correct_prediction'] = valid_df['prediction'] == valid_df['true_label']
        correct_stats = valid_df.groupby('correct_prediction')['saving_ratio'].mean()
        categories = ['Incorrect Predictions', 'Correct Predictions']
        values = [
            correct_stats.get(False, 0),
            correct_stats.get(True, 0)
        ]
        bars = axes[1,1].bar(categories, values, color=['lightcoral', 'lightgreen'])
        axes[1,1].set_ylabel('Average Saving Ratio (%)')
        axes[1,1].set_title('Prediction Accuracy vs Step Saving')
        for i, v in enumerate(values):
            axes[1,1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        
        # 6. Confidence score vs saving ratio
        if 'confidence_score' in valid_df.columns:
            confidence_groups = pd.cut(valid_df['confidence_score'], bins=[0, 3, 6, 9, 10], 
                                     labels=['Low(0-3)', 'Medium(4-6)', 'High(7-9)', 'Very High(9)'])
            saving_by_confidence = valid_df.groupby(confidence_groups)['saving_ratio'].mean()
            saving_by_confidence.plot(kind='bar', color='lightseagreen', ax=axes[1,2])
            axes[1,2].set_ylabel('Average Saving Ratio (%)')
            axes[1,2].set_xlabel('Confidence Score')
            axes[1,2].set_title('Confidence Score vs Step Saving')
            axes[1,2].tick_params(axis='x', rotation=45)
        else:
            axes[1,2].text(0.5, 0.5, 'No Confidence Data', ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Confidence Score vs Step Saving')
        
        plt.tight_layout()
        
        # Save image
        if len(self.models) > 1 and model_name:
            output_file = os.path.join(self.output_dir, f'step_saving_analysis_{model_name}.png')
        else:
            output_file = os.path.join(self.output_dir, 'step_saving_analysis.png')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Step saving analysis plot saved: {output_file}")
        
        # Print step saving statistics
        print(f"\n💾 Step Saving Statistics:")
        print(f"   Overall Saving Ratio: {stats['overall']['saving_ratio']}%")
        print(f"   Total Steps Saved: {stats['overall']['total_steps_saved']:,} steps")
        print(f"   Average Saving Ratio: {stats['overall']['avg_saving_ratio']}%")
        print(f"   Average Steps Saved per Sample: {stats['overall']['avg_steps_saved']:.1f} steps")
        print(f"   Convergent Samples Saving Ratio: {stats['by_true_label']['convergent']['saving_ratio']}%")
        print(f"   Non-convergent Samples Saving Ratio: {stats['by_true_label']['non_convergent']['saving_ratio']}%")
        
        return stats
    
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
        if 'confidence_score' in self.df.columns:
            probabilities = self._get_confidence_scores(self.df)
            if probabilities is not None and len(probabilities) > 0:
                try:
                    mask = ~np.isnan(probabilities)
                    if mask.any():
                        auc_score = roc_auc_score(true_labels[mask], probabilities[mask])
                except Exception as e:
                    if self.debug:
                        print(f"AUC calculation failed: {e}")
        
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
            model_name: Model name, if None use first model
        """
        if model_name is None:
            model_name = self.models[0] if self.models else 'single_model'
        
        # Filter data for specific model
        if 'model' in self.df.columns and model_name != 'single_model':
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return
        else:
            model_df = self.df
        
        true_labels = model_df['true_label'].values
        
        # Get confidence scores
        probabilities = self._get_confidence_scores(model_df)
        
        if probabilities is None:
            print(f"⚠️ Model {model_name} has no confidence values, cannot plot ROC curve")
            return
        
        # Remove NaN values
        mask = ~np.isnan(probabilities)
        if not mask.any():
            print(f"⚠️ Model {model_name} has no valid probability values")
            return
        
        true_labels_clean = true_labels[mask]
        probabilities_clean = probabilities[mask]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels_clean, probabilities_clean)
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
        
        # Save image
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
            Standardized probability value array, or None
        """
        # Check if valid confidence scores exist
        if 'confidence_score' in model_df.columns and validate_confidence_scores(model_df, 'confidence_score'):
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
                if values.notna().any() and len(values) > 0:
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
            model_name = self.models[0] if self.models else 'single_model'
        
        if 'model' in self.df.columns and model_name != 'single_model':
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
        
        # Save image
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
            model_name = self.models[0] if self.models else 'single_model'
        
        if 'model' in self.df.columns and model_name != 'single_model':
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
        
        # Remove NaN values
        mask = ~np.isnan(probabilities)
        if not mask.any():
            print(f"⚠️ Model {model_name} has no valid probability values")
            return
        
        true_labels_clean = true_labels[mask]
        probabilities_clean = probabilities[mask]
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(true_labels_clean, probabilities_clean)
        avg_precision = average_precision_score(true_labels_clean, probabilities_clean)
        
        # Plot PR curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        
        # Add baseline (random classifier performance)
        positive_ratio = np.mean(true_labels_clean)
        plt.axhline(y=positive_ratio, color='red', linestyle='--', 
                   label=f'Random classifier (AP = {positive_ratio:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        if len(self.models) > 1:
            plt.title(f'{model_name} - Precision-Recall Curve', fontsize=14)
        else:
            plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        # Save image
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
            model_name = self.models[0] if self.models else 'single_model'
        
        if 'model' in self.df.columns and model_name != 'single_model':
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
        
        # Save image
        if len(self.models) > 1:
            output_file = os.path.join(self.output_dir, f'confidence_distribution_{model_name}.png')
        else:
            output_file = os.path.join(self.output_dir, 'confidence_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confidence distribution plot saved: {output_file}")
        
        # Print distribution statistics
        print("Confidence Score Distribution:")
        for score, count in confidence_scores.items():
            print(f"   {score}: {count} samples")
        
        return confidence_scores.to_dict()
    
    def analyze_model(self, model_name: str = None) -> Dict:
        """
        Analyze single model performance
        
        Args:
            model_name: Model name
            
        Returns:
            Analysis results dictionary
        """
        if model_name is None:
            model_name = self.models[0] if self.models else 'single_model'
        
        if 'model' in self.df.columns and model_name != 'single_model':
            model_df = self.df[self.df['model'] == model_name]
            if len(model_df) == 0:
                print(f"❌ No data found for model {model_name}")
                return {}
        else:
            model_df = self.df
        
        if len(self.models) > 1:
            print(f"\n📊 Analyzing model: {model_name}")
        else:
            print(f"\n📊 Analysis Results")
        print(f"Sample count: {len(model_df)}")
        
        # Calculate basic metrics
        predictions = model_df['prediction'].values
        true_labels = model_df['true_label'].values
        
        metrics = self.calculate_metrics(predictions, true_labels)
        
        # Print basic metrics
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print(f"Specificity: {metrics['specificity']:.3f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        
        if metrics['auc'] is not None:
            print(f"AUC: {metrics['auc']:.3f}")
        
        # Plot charts
        self.plot_roc_curve(model_name)
        self.plot_confusion_matrix(model_name)
        self.plot_precision_recall_curve(model_name)
        self.plot_confidence_distribution(model_name)
        
        # Step saving analysis (if step data exists)
        step_stats = self.plot_step_saving_analysis(model_name)
        if step_stats:
            metrics['step_saving'] = step_stats
        
        return metrics
    
    def generate_report(self):
        """
        Generate complete analysis report
        """
        print(f"\n📊 Generating Complete Analysis Report")
        print("=" * 80)
        
        # Read original data to get total sample count
        try:
            original_df = pd.read_csv(self.results_file, encoding='utf-8-sig')
            total_original_samples = len(original_df)
        except:
            total_original_samples = len(self.df)
        
        report = {
            'data_info': {
                'total_original_samples': total_original_samples,
                'analyzed_samples': len(self.df),
                'success_rate': len(self.df) / total_original_samples if total_original_samples > 0 else 1.0,
                'models': self.models,
            },
            'model_results': {}
        }
        
        # Analyze each model
        for model in self.models:
            print(f"\nAnalyzing model: {model}")
            metrics = self.analyze_model(model)
            report['model_results'][model] = metrics
        
        # If single model, also analyze
        if len(self.models) == 1 and self.models[0] == 'single_model':
            metrics = self.analyze_model('single_model')
            report['model_results']['single_model'] = metrics
        
        # Save report
        report_file = os.path.join(self.output_dir, 'analysis_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Complete analysis report saved: {report_file}")
        
        # Print step saving summary (if exists)
        step_stats = self.calculate_step_saving()
        if step_stats:
            print(f"\n💾 Step Saving Summary:")
            overall = step_stats['overall']
            print(f"   Overall Saving Ratio: {overall['saving_ratio']}%")
            print(f"   Total Steps Saved: {overall['total_steps_saved']:,} steps")
            print(f"   Average Saving Ratio: {overall['avg_saving_ratio']}%")
            print(f"   Average Steps Saved per Sample: {overall['avg_steps_saved']:.1f} steps")
            print(f"   Convergent Samples Saving Ratio: {step_stats['by_true_label']['convergent']['saving_ratio']}%")
            print(f"   Non-convergent Samples Saving Ratio: {step_stats['by_true_label']['non_convergent']['saving_ratio']}%")
        
        print(f"\n🎉 Analysis completed! All results saved in: {self.output_dir}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Results Analysis')
    parser.add_argument('results_file', help='Path to evaluation results file')
    parser.add_argument('--output-dir', '-o', default=None, 
                       help='Output directory, if None automatically generated based on input filename')
    parser.add_argument('--model', '-m', help='Specify model name to analyze')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_file, args.output_dir, args.debug)
    
    if analyzer.df is None or len(analyzer.df) == 0:
        print("❌ Data loading failed, cannot perform analysis")
        return
    
    # Execute analysis based on parameters
    if args.model:
        analyzer.analyze_model(args.model)
    else:
        analyzer.generate_report()


if __name__ == "__main__":
    main()
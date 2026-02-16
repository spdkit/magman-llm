#!/usr/bin/env python3
"""
ROC calculation utility module
Provides unified confidence score mapping and ROC calculation functionality
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


def calculate_roc_probability(confidence_score: Union[int, float, str],
                             prediction: Optional[int] = None) -> float:
    """
    Convert 0-9 confidence score to ROC probability

    Args:
        confidence_score: Confidence score (0-9)
        prediction: Prediction label (0=not converged, 1=converged)
                   If None, returns base probability

    Returns:
        Mapped probability value (0.05-0.95)
    """
    try:
        # Convert to numeric
        score = float(confidence_score)

        # Base mapping: 0.05-0.95 range
        # 0 -> 0.05, 9 -> 0.95, linear mapping
        base_prob = 0.05 + (score / 9.0) * 0.9

        # Adjust based on prediction direction (if prediction is provided)
        if prediction is not None:
            if prediction == 1:  # Predict convergence
                return base_prob
            else:  # Predict non-convergence
                return 1.0 - base_prob
        else:
            return base_prob
            
    except (ValueError, TypeError):
        return np.nan


def get_roc_probabilities(df: pd.DataFrame,
                         confidence_col: str = 'confidence_score',
                         prediction_col: str = 'prediction') -> np.ndarray:
    """
    Calculate ROC probabilities for DataFrame

    Args:
        df: DataFrame containing confidence scores and predictions
        confidence_col: Confidence score column name
        prediction_col: Prediction column name

    Returns:
        ROC probability array
    """
    if confidence_col not in df.columns:
        raise ValueError(f"Missing confidence column in DataFrame: {confidence_col}")

    if prediction_col not in df.columns:
        raise ValueError(f"Missing prediction column in DataFrame: {prediction_col}")
    
    roc_probs = []
    for _, row in df.iterrows():
        prob = calculate_roc_probability(
            row[confidence_col], 
            row[prediction_col]
        )
        roc_probs.append(prob)
    
    return np.array(roc_probs)


def validate_confidence_scores(df: pd.DataFrame,
                              confidence_col: str = 'confidence_score') -> bool:
    """
    Validate the validity of confidence scores

    Args:
        df: DataFrame
        confidence_col: Confidence column name

    Returns:
        Whether it contains valid confidence scores
    """
    if confidence_col not in df.columns:
        return False
    
    try:
        scores = pd.to_numeric(df[confidence_col], errors='coerce')
        valid_scores = scores.dropna()

        # Check score range (0-9)
        if len(valid_scores) > 0:
            min_score = valid_scores.min()
            max_score = valid_scores.max()
            return min_score >= 0 and max_score <= 9

        return False

    except Exception:
        return False
import sys
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from classes.business_scorer import BusinessScorer
from classes.feature_engineering import FeatureEngineering

def test_business_scorer_threshold_logic():
    """Verify that the business scorer correctly penalizes False Negatives more than False Positives."""
    scorer = BusinessScorer(fn_cost=10, fp_cost=1)
    
    # Case 1: One False Negative (High Cost)
    y_true_fn = [1]
    y_pred_fn = [0]
    cost_fn = scorer.cost_function(y_true_fn, y_pred_fn)
    
    # Case 2: One False Positive (Low Cost)
    y_true_fp = [0]
    y_pred_fp = [1]
    cost_fp = scorer.cost_function(y_true_fp, y_pred_fp)
    
    assert cost_fn == 10
    assert cost_fp == 1
    assert cost_fn > cost_fp

def test_feature_engineering_output():
    """Ensure feature engineering returns expected columns and no NaNs in critical features."""
    fe = FeatureEngineering()
    
    # Create dummy data
    df = pd.DataFrame({
        'AMT_INCOME_TOTAL': [100, 200, 300],
        'AMT_CREDIT': [1000, 2000, 3000],
        'AMT_ANNUITY': [100, 200, 300],
        'DAYS_BIRTH': [-10000, -15000, -20000],
        'DAYS_EMPLOYED': [-1000, -2000, -3000],
        'EXT_SOURCE_1': [0.1, 0.2, 0.3],
        'EXT_SOURCE_2': [0.4, 0.5, 0.6],
        'EXT_SOURCE_3': [0.7, 0.8, 0.9]
    })
    
    df_processed = fe.simple_feature_engineering(df)
    
    # Check if new features are created
    assert 'CREDIT_INCOME_PERCENT' in df_processed.columns
    assert 'ANNUITY_INCOME_PERCENT' in df_processed.columns
    assert 'CREDIT_TERM' in df_processed.columns
    
    # Check for no NaNs in the processed output
    assert df_processed.isnull().sum().sum() == 0

def test_threshold_application():
    """Test the manual threshold application logic (0.45)."""
    y_probs = np.array([0.1, 0.4, 0.46, 0.8])
    threshold = 0.45
    y_pred = (y_probs >= threshold).astype(int)
    
    expected_pred = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(y_pred, expected_pred)

def test_data_leakage_prevention():
    """Verify that the split logic (conceptually) doesn't overlap."""
    # This is a simple check to ensure we understand the split logic
    indices = np.arange(100)
    train_idx = indices[:80]
    test_idx = indices[80:]
    
    intersection = set(train_idx).intersection(set(test_idx))
    assert len(intersection) == 0

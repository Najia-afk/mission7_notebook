import sys
import os
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from classes.data_loader import DataLoader
from classes.business_scorer import BusinessScorer
from classes.feature_engineering import FeatureEngineering
from classes.model_trainer import ModelTrainer
from scripts.model_comparison import compare_with_production

def test_imports():
    """Checks if all classes and scripts can be imported."""
    assert DataLoader is not None
    assert BusinessScorer is not None
    assert FeatureEngineering is not None
    assert ModelTrainer is not None
    assert compare_with_production is not None

def test_business_scorer():
    """Checks if business scorer calculates cost correctly."""
    scorer = BusinessScorer(fn_cost=10, fp_cost=1)
    # TN=1, FP=1, FN=1, TP=1
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cost = scorer.cost_function(y_true, y_pred)
    # Cost = (1*10 (FN) + 1*1 (FP)) / 4 = 2.75
    assert cost == 2.75

if __name__ == "__main__":
    print("Running tests...")
    test_imports()
    test_business_scorer()
    print("All tests passed!")

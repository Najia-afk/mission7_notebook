"""
Script: Model Evaluation & Selection (Step 8)
Evaluates multiple models, selects best, and tests on fresh data.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


def evaluate_and_select_models(models_dict, X_val, y_val, X_test_final, y_test_final, business_scorer):
    """
    Evaluate multiple models on validation set, select best, then test on fresh data.
    
    Args:
        models_dict: Dict of {name: model_object}
        X_val, y_val: Validation data
        X_test_final, y_test_final: Fresh test data
        business_scorer: BusinessScorer object with cost_function()
    
    Returns:
        dict with results, best_model_name, and metrics
    """
    # ===== VALIDATION SET EVALUATION =====
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        results.append({
            'Model': name,
            'Business Cost (Avg)': business_scorer.cost_function(y_val, y_pred),
            'AUC': roc_auc_score(y_val, y_proba),
            'F1-Score': f1_score(y_val, y_pred)
        })
    
    # Create leaderboard and select best model
    leaderboard = pd.DataFrame(results).sort_values(by='Business Cost (Avg)')
    best_model_name = leaderboard.iloc[0]['Model']
    best_model = models_dict[best_model_name]
    
    # ===== FRESH TEST SET EVALUATION =====
    y_test_pred = best_model.predict(X_test_final)
    y_test_proba = best_model.predict_proba(X_test_final)[:, 1]
    
    test_cost = business_scorer.cost_function(y_test_final, y_test_pred)
    test_auc = roc_auc_score(y_test_final, y_test_proba)
    test_f1 = f1_score(y_test_final, y_test_pred)
    
    # Get validation metrics
    val_cost = leaderboard[leaderboard['Model'] == best_model_name]['Business Cost (Avg)'].values[0]
    val_auc = leaderboard[leaderboard['Model'] == best_model_name]['AUC'].values[0]
    val_f1 = leaderboard[leaderboard['Model'] == best_model_name]['F1-Score'].values[0]
    
    return {
        'leaderboard': leaderboard,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'val_cost': val_cost,
        'val_auc': val_auc,
        'val_f1': val_f1,
        'test_cost': test_cost,
        'test_auc': test_auc,
        'test_f1': test_f1,
    }


def print_evaluation_summary(results):
    """Print clean evaluation summary."""
    print("\n" + "="*70)
    print("Model Evaluation & Selection")
    print("="*70)
    print(f"VALIDATION SET (used for model selection):\n")
    
    leaderboard = results['leaderboard']
    print(leaderboard.to_string(index=False))
    
    print(f"\nSelected Best Model: {results['best_model_name']}")
    
    print(f"\n{'='*70}")
    print(f"FINAL CHALLENGE: Testing on Fresh Test Set")
    print(f"{'='*70}")
    print(f"This data has not been seen during training or hyperparameter tuning!\n")
    
    print(f"VALIDATION SET METRICS:")
    print(f"   Business Cost (Avg): {results['val_cost']:.4f}")
    print(f"   AUC:                 {results['val_auc']:.4f}")
    print(f"   F1-Score:            {results['val_f1']:.4f}")
    
    print(f"\nTEST SET METRICS:")
    print(f"   Business Cost (Avg): {results['test_cost']:.4f}")
    print(f"   AUC:                 {results['test_auc']:.4f}")
    print(f"   F1-Score:            {results['test_f1']:.4f}")
    
    print(f"\nGENERALIZATION GAP (Val vs Test):")
    print(f"   Cost difference:     {abs(results['test_cost'] - results['val_cost']):.4f}")
    print(f"   AUC difference:      {abs(results['test_auc'] - results['val_auc']):.4f}")
    print(f"   F1 difference:       {abs(results['test_f1'] - results['val_f1']):.4f}")
    print("="*70 + "\n")

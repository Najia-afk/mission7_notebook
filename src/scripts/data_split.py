"""
Script: Data Split Setup (Step 6.5)
Handles 3-way data split: Train+Val | Test_Final
"""
from sklearn.model_selection import train_test_split

def create_data_splits(X, y, test_size=0.2, random_state=42):
    """
    Create 3-way data split: Train | Val | Test_Final
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction for test set (default 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        dict with all splits and their shapes
    """
    # FIRST SPLIT: Reserve test_size % of FULL data as untouched test set
    X_train_val, X_test_final, y_train_val, y_test_final = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # SECOND SPLIT: Split train_val into Train (80%) and Val (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=random_state, stratify=y_train_val
    )
    
    # Return splits
    splits = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test_final': X_test_final,
        'y_test_final': y_test_final,
        'X_train_val': X_train_val,
        'y_train_val': y_train_val,
    }
    
    return splits


def print_split_summary(splits):
    """Print a clean summary of the data splits."""
    print("\n" + "="*70)
    print("Data Split Summary")
    print("="*70)
    print(f"Training set:      {splits['X_train'].shape[0]:>8,} samples ({splits['X_train'].shape[0]/len(splits['X_train_val'])*100:.1f}%)")
    print(f"Validation set:    {splits['X_val'].shape[0]:>8,} samples ({splits['X_val'].shape[0]/len(splits['X_train_val'])*100:.1f}%)")
    print(f"Test_Final set:    {splits['X_test_final'].shape[0]:>8,} samples (20% - RESERVED)")
    print(f"\nTest_Final will NOT be touched until final evaluation!")
    print("="*70 + "\n")

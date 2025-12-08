import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'TARGET'):
    """
    Plots the distribution of the target variable.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title('Target Variable Distribution')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the top N feature importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    plt.show()

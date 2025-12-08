import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix

class BusinessScorer:
    """
    Custom scorer for the business problem.
    Penalizes False Negatives (FN) more than False Positives (FP).
    """
    def __init__(self, fn_cost: float = 10.0, fp_cost: float = 1.0):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def cost_function(self, y_true, y_pred):
        """
        Calculates the total cost based on the confusion matrix.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (fn * self.fn_cost) + (fp * self.fp_cost)
        return cost

    def get_scorer(self):
        """
        Returns a sklearn scorer object. 
        Note: sklearn scorers maximize the score, so we return negative cost.
        """
        return make_scorer(self.cost_function, greater_is_better=False)

    def calculate_optimal_threshold(self, y_true, y_proba):
        """
        Finds the optimal probability threshold that minimizes the cost.
        """
        thresholds = np.linspace(0, 1, 101)
        costs = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            costs.append(self.cost_function(y_true, y_pred))
            
        best_threshold = thresholds[np.argmin(costs)]
        min_cost = min(costs)
        
        return best_threshold, min_cost

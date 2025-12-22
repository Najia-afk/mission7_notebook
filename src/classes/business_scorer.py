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
        Calculates the average cost per client based on the confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases where only one class is predicted
            tn = fp = fn = tp = 0
            if len(np.unique(y_true)) == 1:
                if y_true[0] == 0: fp = np.sum(y_pred == 1)
                else: fn = np.sum(y_pred == 0)
            else:
                if np.unique(y_pred)[0] == 0: fn = np.sum(y_true == 1)
                else: fp = np.sum(y_true == 0)

        cost = (fn * self.fn_cost) + (fp * self.fp_cost)
        return cost / len(y_true) if len(y_true) > 0 else 0.0

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

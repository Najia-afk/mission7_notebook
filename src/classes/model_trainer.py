import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
import warnings

class ModelTrainer:
    """
    Class to handle model training, tuning, and MLflow logging.
    """
    def __init__(self, experiment_name: str = "HomeCredit_DefaultRisk"):
        mlflow.set_experiment(experiment_name)

    def train_and_log(self, pipeline: ImbPipeline, param_grid: dict, X_train, y_train, scorer, run_name: str, step_name: str = "model_training"):
        """
        Trains a model using GridSearchCV and logs results to MLflow.
        """
        with mlflow.start_run(run_name=run_name):
            # Log parameter grid as a tag or artifact instead of params to avoid conflict
            mlflow.set_tag("param_grid", str(param_grid))
            
            # Cross-validation strategy
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # Grid Search
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scorer,
                cv=cv,
                n_jobs=-1,
                verbose=0  # Reduce verbosity to avoid spam
            )
            
            print(f"Starting Grid Search for {run_name}...")
            
            # Suppress specific warnings during fitting
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
                grid_search.fit(X_train, y_train)
            
            # Log best metrics and params
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            
            print(f"Best Score: {best_score}")
            print(f"Best Params: {best_params}")
            
            mlflow.log_metric("best_cv_score", best_score)
            mlflow.log_params(best_params)
            
            # Log detailed CV results
            cv_results = grid_search.cv_results_
            for i in range(len(cv_results['params'])):
                params_str = str(cv_results['params'][i])
                mean_score = cv_results['mean_test_score'][i]
                std_score = cv_results['std_test_score'][i]
                mlflow.log_metric(f"cv_mean_score_{i}", mean_score)
                mlflow.log_metric(f"cv_std_score_{i}", std_score)
                # Log params as text to correlate with index if needed, or rely on params logged above
            
            # Log the best model
            # Fix: Use artifact_path explicitly to avoid warning
            mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="model")
            
            # Log step tag
            mlflow.set_tag("step", step_name)
            
            return grid_search.best_estimator_

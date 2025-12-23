import mlflow
import mlflow.sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

class ModelTrainer:
    """
    Class to handle model training, tuning, and MLflow logging.
    """
    def __init__(self, experiment_name: str = None):
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    def train_and_log(self, pipeline: ImbPipeline, param_grid: dict, X_train, y_train, scorer, run_name: str, step_name: str = "model_training", factor: int = 5, n_jobs: int = -1):
        """
        Trains a model using HalvingGridSearchCV (Successive Halving) for professional pruning.
        Logs all candidate trials to MLflow.
        
        Parameters:
        -----------
        factor : int, default=5
            Factor for successive halving (5=aggressive, 4=moderate, 3=conservative)
        n_jobs : int, default=-1
            Number of parallel jobs to run. -1 means using all processors.
        """
        with mlflow.start_run(run_name=run_name) as parent_run:
            mlflow.set_tag("param_grid", str(param_grid))
            mlflow.set_tag("search_type", "HalvingGridSearchCV")
            mlflow.set_tag("pruning_factor", factor)
            mlflow.set_tag("n_jobs", n_jobs)
            
            # Cross-validation strategy
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # Halving Grid Search (Successive Halving)
            search = HalvingGridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scorer,
                cv=cv,
                factor=factor,
                resource='n_samples',
                min_resources=500,  # Ensure enough samples for SMOTE
                random_state=42,
                n_jobs=n_jobs,
                verbose=0
            )
            
            print(f"Starting Halving Search for {run_name}...")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                search.fit(X_train, y_train)
            
            # Log best metrics and params to parent run
            best_score = search.best_score_
            best_params = search.best_params_
            
            print(f"Best Score: {best_score}")
            print(f"Best Params: {best_params}")
            
            mlflow.log_metric("best_cv_score", best_score)
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")
            mlflow.set_tag("step", step_name)
            
            return search

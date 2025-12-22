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
    def __init__(self, experiment_name: str = "HomeCredit_DefaultRisk"):
        mlflow.set_experiment(experiment_name)

    def train_and_log(self, pipeline: ImbPipeline, param_grid: dict, X_train, y_train, scorer, run_name: str, step_name: str = "model_training"):
        """
        Trains a model using HalvingGridSearchCV (Successive Halving) for professional pruning.
        Logs all candidate trials to MLflow.
        """
        with mlflow.start_run(run_name=run_name) as parent_run:
            mlflow.set_tag("param_grid", str(param_grid))
            mlflow.set_tag("search_type", "HalvingGridSearchCV")
            
            # Cross-validation strategy
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # Halving Grid Search (Successive Halving)
            search = HalvingGridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scorer,
                cv=cv,
                factor=3,
                resource='n_samples',
                min_resources=500,  # Ensure enough samples for SMOTE
                random_state=42,
                n_jobs=16, # Optimized for i9-14900K (32 logical processors)
                verbose=0
            )
            
            print(f"Starting Professional Halving Search for {run_name} (Pruning enabled)...")
            
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

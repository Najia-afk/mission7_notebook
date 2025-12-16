import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create indicator columns for missing values.
    1 = Original (Not Missing), 0 = Imputed (Missing)
    """
    def __init__(self, suffix='_origin', return_only_indicators=False):
        self.suffix = suffix
        self.return_only_indicators = return_only_indicators
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        # Handle pandas DataFrame
        if hasattr(X, 'iloc'):
            if self.return_only_indicators:
                X_indicators = pd.DataFrame(index=X.index)
                for col in X.columns:
                    X_indicators[f"{col}{self.suffix}"] = X[col].notna().astype(int)
                return X_indicators
            else:
                X_new = X.copy()
                for col in X.columns:
                    # 1 if not NaN (original), 0 if NaN (imputed)
                    X_new[f"{col}{self.suffix}"] = X[col].notna().astype(int)
                return X_new
        
        # Handle numpy array
        else:
            # Assuming X is 2D array
            X_indicators = (~np.isnan(X)).astype(int)
            
            if self.return_only_indicators:
                return X_indicators
                
            return np.hstack([X, X_indicators])
            
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        
        if input_features is None:
             return None 
             
        if self.return_only_indicators:
            output_features = []
            for f in input_features:
                output_features.append(f"{f}{self.suffix}")
            return np.array(output_features)
            
        output_features = list(input_features)
        for f in input_features:
            output_features.append(f"{f}{self.suffix}")
        return np.array(output_features)

class FeatureEngineering:
    """
    Class to handle feature engineering and preprocessing.
    """
    def __init__(self):
        self.preprocessor = None

    def create_preprocessor(self, numeric_features, categorical_features, use_knn=False):
        """
        Creates a sklearn ColumnTransformer for preprocessing.
        """
        if use_knn:
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy='median')

        numeric_transformer = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', StandardScaler())
        ])
        
        indicator_transformer = MissingIndicatorTransformer(return_only_indicators=True)

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('ind', indicator_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor

    def simple_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs simple feature engineering like creating ratios.
        """
        df = df.copy()
        
        # Example domain knowledge features
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        
        return df

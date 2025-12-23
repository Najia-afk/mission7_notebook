import sys
import os
import pandas as pd
import numpy as np
import mlflow
import shap
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) # src
project_dir = os.path.dirname(src_dir) # mission7_notebook
sys.path.insert(0, src_dir)

from classes.data_loader import DataLoader
from classes.sqlite_connector import DatabaseConnection
from classes.feature_engineering import FeatureEngineering

# Configuration
DATA_PATH = os.path.join(project_dir, 'dataset')
DB_PATH = os.path.join(DATA_PATH, 'home_credit.db')
RUN_ID = "90ea9ec38c4e4b94be906c2713242313"
MLFLOW_URI = "http://mlflow:5005"

def main():
    print("🚀 Starting Feature Extraction Script...")
    
    # 1. Load Data
    print(f"📂 Loading data from {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    db = DatabaseConnection(DB_PATH)
    # Load full table to ensure we have all columns for FE
    try:
        df_train = db.read_table('application_train')
        print(f"   Data loaded: {df_train.shape}")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # 2. Feature Engineering (Monkey Mode)
    print("🛠️ Applying Feature Engineering...")
    fe = FeatureEngineering()
    df_train = fe.simple_feature_engineering(df_train)
    
    # Prepare X (Monkey Mode = All features except Target/ID)
    cols_to_exclude = ['TARGET', 'SK_ID_CURR']
    feature_cols = [c for c in df_train.columns if c not in cols_to_exclude]
    X = df_train[feature_cols]
    
    print(f"   Feature set prepared: {X.shape[1]} features")

    # 3. Load Model
    print(f"🔄 Loading model from MLflow (Run ID: {RUN_ID})...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"runs:/{RUN_ID}/model"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return

    # 4. Compute SHAP
    print("📊 Computing SHAP values (using 500 samples)...")
    # Sample X for SHAP
    X_sample = X.sample(n=500, random_state=42)
    
    # Extract steps from Pipeline
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform data
        print("   Preprocessing data...")
        X_transformed = preprocessor.transform(X_sample)
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
        # Convert to DF
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Explain
        print("   Explaining model...")
        # TreeExplainer for LGBM
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed_df)
        
        # Handle SHAP output (list for binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Positive class
            
        # Calculate Mean Absolute SHAP
        vals = np.abs(shap_values).mean(0)
        
        # Create DataFrame
        feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        
        # 5. Output Top 30
        top_30 = feature_importance.head(30)['col_name'].tolist()
        
        print("\n" + "="*50)
        print("🏆 TOP 30 FEATURES")
        print("="*50)
        # Print as a clean python list string
        print(json.dumps(top_30, indent=4))
        print("="*50)
        
        # Save to file
        output_path = os.path.join(src_dir, 'scripts', 'top_30_features.json')
        with open(output_path, 'w') as f:
            json.dump(top_30, f, indent=4)
        print(f"\nSaved list to {output_path}")
        
    else:
        print("   ❌ Model structure not recognized (not a Pipeline?)")

if __name__ == "__main__":
    main()

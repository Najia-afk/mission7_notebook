import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classes.sqlite_connector import DatabaseConnection
from src.classes.model_visualizer import ModelVisualizer
from src.classes.feature_engineering import FeatureEngineering

app = Flask(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5005")
DB_PATH = os.getenv("DB_PATH", "/app/dataset/home_credit.db")
MODEL_NAME = "CreditScoring_BestModel"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Global variable to store column names
ALL_COLUMNS = None

def get_all_columns():
    """Retrieves all column names from the database to use as a template."""
    global ALL_COLUMNS
    if ALL_COLUMNS is not None:
        return ALL_COLUMNS
    
    try:
        db = DatabaseConnection(DB_PATH)
        # Just get 1 row to see the columns
        df = db.execute_query("SELECT * FROM application_train LIMIT 1")
        ALL_COLUMNS = df.columns.tolist()
        return ALL_COLUMNS
    except Exception as e:
        print(f"Error getting columns: {e}")
        return []

def get_production_model():
    """Loads the model tagged as 'Production' from MLflow."""
    try:
        # Load production model
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get threshold from run params
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            run = client.get_run(run_id)
            threshold = float(run.data.params.get("business_optimal_threshold", 0.45))
        else:
            threshold = 0.45
            
        return model, threshold
    except Exception as e:
        print(f"Error loading production model: {e}")
        return None, 0.45

@app.route('/')
@app.route('/api/html')
def index():
    return render_template('index.html')

@app.route('/api/client/<int:client_id>')
def get_client_data(client_id):
    """Fetches raw client data from the database."""
    try:
        db = DatabaseConnection(DB_PATH)
        query = f"SELECT * FROM application_train WHERE SK_ID_CURR = {client_id}"
        df_client = db.execute_query(query)
        
        if df_client.empty:
            query = f"SELECT * FROM application_test WHERE SK_ID_CURR = {client_id}"
            df_client = db.execute_query(query)
            
        if df_client.empty:
            return jsonify({"error": f"Client {client_id} not found"}), 404
            
        # Convert to dict, handle NaN for JSON compatibility
        data = df_client.iloc[0].replace({np.nan: None}).to_dict()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Support both form data (from existing UI) and JSON (for what-if analysis)
    if request.is_json:
        data = request.get_json()
        client_id = data.get('client_id')
        manual_features = data.get('features')
    else:
        client_id = request.form.get('client_id')
        manual_features = None
        
    if not client_id and not manual_features:
        return jsonify({"error": "No Client ID or features provided"}), 400
    
    try:
        # 1. Load/Prepare Data
        if manual_features:
            # Use provided features directly, but fill missing columns with NaN
            all_cols = get_all_columns()
            # Create a template with all columns as NaN
            df_template = pd.DataFrame(columns=all_cols)
            # Add one row of NaNs
            df_template.loc[0] = [np.nan] * len(all_cols)
            
            # Update with manual features
            for key, value in manual_features.items():
                if key in df_template.columns:
                    df_template.at[0, key] = value
            
            df_client = df_template
        else:
            # Fetch from DB
            db = DatabaseConnection(DB_PATH)
            query = f"SELECT * FROM application_train WHERE SK_ID_CURR = {client_id}"
            df_client = db.execute_query(query)
            
            if df_client.empty:
                query = f"SELECT * FROM application_test WHERE SK_ID_CURR = {client_id}"
                df_client = db.execute_query(query)
                
            if df_client.empty:
                return jsonify({"error": f"Client {client_id} not found in database."}), 404
            
        # 2. Preprocess (Feature Engineering)
        fe = FeatureEngineering()
        df_processed = fe.simple_feature_engineering(df_client)
        
        # 3. Load Model
        model, threshold = get_production_model()
        if model is None:
            return jsonify({"error": "Production model not found in MLflow Registry."}), 500
            
        # 4. Predict
        # Drop target if exists
        X = df_processed.drop(columns=['TARGET'], errors='ignore')
        # Ensure SK_ID_CURR is not in features
        X_features = X.drop(columns=['SK_ID_CURR'], errors='ignore')
        
        y_proba = model.predict_proba(X_features)[:, 1][0]
        decision = "REJECTED" if y_proba >= threshold else "ACCEPTED"
        
        # 5. SHAP Explanation
        visualizer = ModelVisualizer()
        shap_data = visualizer.compute_shap_values(model, X_features, sample_size=1)
        
        if shap_data:
            fig_local = visualizer.plot_shap_local(shap_data, sample_idx=0)
            shap_html = fig_local.to_html(full_html=False, include_plotlyjs=False)
        else:
            shap_html = "<p>SHAP explanation not available for this model type.</p>"
            
        return jsonify({
            "client_id": client_id or "Manual Input",
            "probability": float(y_proba),
            "threshold": threshold,
            "decision": decision,
            "shap_html": shap_html
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

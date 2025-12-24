import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.classes.sqlite_connector import DatabaseConnection
import os

def generate_drift_report(db_path, output_path):
    print(f"Loading data from {db_path}...")
    db = DatabaseConnection(db_path)
    
    # Load reference data (train)
    ref_df = db.execute_query("SELECT * FROM application_train LIMIT 1000")
    # Load current data (test)
    curr_df = db.execute_query("SELECT * FROM application_test LIMIT 1000")
    
    print("Generating Evidently Drift Report...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])
    
    # Drop ID and Target for drift analysis if they exist
    ref_features = ref_df.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    curr_features = curr_df.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    
    # Ensure columns match
    common_cols = list(set(ref_features.columns) & set(curr_features.columns))
    ref_features = ref_features[common_cols]
    curr_features = curr_features[common_cols]
    
    snapshot = report.run(reference_data=ref_features, current_data=curr_features)
    
    print(f"Saving report to {output_path}...")
    snapshot.save_html(output_path)
    print("Report generated successfully.")

if __name__ == "__main__":
    DB_PATH = os.getenv("DB_PATH", "dataset/home_credit.db")
    OUTPUT_PATH = "evidently_drift_report.html"
    generate_drift_report(DB_PATH, OUTPUT_PATH)

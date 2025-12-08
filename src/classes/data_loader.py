import pandas as pd
import os

class DataLoader:
    """
    Class to load and manage the Home Credit Default Risk dataset.
    """
    def __init__(self, data_path: str = "dataset"):
        self.data_path = data_path

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Loads a CSV file from the data directory.
        """
        file_path = os.path.join(self.data_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading {file_name}...")
        return pd.read_csv(file_path)

    def load_train_data(self) -> pd.DataFrame:
        """Loads the main training data."""
        return self.load_data("application_train.csv")

    def load_test_data(self) -> pd.DataFrame:
        """Loads the main test data."""
        return self.load_data("application_test.csv")

    def load_all_tables(self) -> dict:
        """
        Loads all relevant tables into a dictionary.
        """
        files = [
            "application_train.csv", "application_test.csv", "bureau.csv", 
            "bureau_balance.csv", "credit_card_balance.csv", 
            "installments_payments.csv", "POS_CASH_balance.csv", 
            "previous_application.csv"
        ]
        data = {}
        for f in files:
            try:
                data[f.replace(".csv", "")] = self.load_data(f)
            except FileNotFoundError:
                print(f"Warning: {f} not found, skipping.")
        return data

    def create_database(self, db_path: str = "dataset/home_credit.db"):
        """
        Creates a SQLite database from the CSV files.
        """
        from sqlalchemy import create_engine
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        engine = create_engine(f'sqlite:///{db_path}')
        
        data = self.load_all_tables()
        for table_name, df in data.items():
            print(f"Writing {table_name} to database...")
            df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"Database created at {db_path}")


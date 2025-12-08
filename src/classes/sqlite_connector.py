from typing import Optional, List
import os
import pandas as pd
from sqlalchemy import create_engine, Engine, inspect
from pathlib import Path

class DatabaseConnection:
    """Database connection manager with caching capabilities."""
    
    def __init__(self, db_path: str) -> None:
        """Initialize database connection and cache directory."""
        self.db_path = db_path
        self._engine: Optional[Engine] = None
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        script_dir = Path(__file__).resolve().parent
        self.cache_dir = script_dir.parent.parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)

    def get_engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(f'sqlite:///{self.db_path}')
        return self._engine

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        sanitized_key = "".join(c for c in key if c.isalnum() or c in ('-', '_'))
        return self.cache_dir / f"{sanitized_key}.pkl"

    def _save_to_cache(self, df: pd.DataFrame, key: str) -> None:
        """Save DataFrame to cache."""
        try:
            cache_path = self._get_cache_path(key)
            df.to_pickle(str(cache_path))
        except Exception as e:
            print(f"Warning: Failed to save cache for {key}: {e}")

    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if exists."""
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                return pd.read_pickle(str(cache_path))
        except Exception as e:
            print(f"Warning: Failed to load cache for {key}: {e}")
        return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def get_table_names(self) -> List[str]:
        """Get all table names from database."""
        inspector = inspect(self.get_engine())
        return inspector.get_table_names()

    def read_table(self, table_name: str, use_cache: bool = True) -> pd.DataFrame:
        """Read table into DataFrame with optional caching."""
        if use_cache:
            cached_df = self._load_from_cache(table_name)
            if cached_df is not None:
                return cached_df
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.get_engine())
        if use_cache:
            self._save_to_cache(df, table_name)
        return df

    def execute_query(self, query: str, cache_key: Optional[str] = None) -> pd.DataFrame:
        """Execute SQL query with optional caching."""
        if cache_key:
            cached_df = self._load_from_cache(cache_key)
            if cached_df is not None:
                return cached_df
        
        df = pd.read_sql_query(query, self.get_engine())
        if cache_key:
            self._save_to_cache(df, cache_key)
        return df

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._engine:
            self._engine.dispose()

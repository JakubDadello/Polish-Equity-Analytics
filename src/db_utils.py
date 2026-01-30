import os
import pandas as pd
from sqlalchemy import create_engine

def get_engine():
    # --- Create a SQLAlchemy engine for database connection ---
    db_url = os.getenv("PROJECT_DB")
    if not db_url:
        raise ValueError(f"Environment variable {db_url} not found")
    return create_engine(db_url)

def load_csv_to_db (csv_path: str, table_name: str = "raw_data"):
    engine = get_engine()
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def load_data(query: str) -> pd.DataFrame:
    """
    Execute SQL query and return results as a pandas DataFrame.
    
    Args:
        query: SQL string to be executed.
    Returns:
        DataFrame containing the query results.
    """
    engine = get_engine()
    with engine.connect() as connection:
        return pd.read_sql(query, connection)
 

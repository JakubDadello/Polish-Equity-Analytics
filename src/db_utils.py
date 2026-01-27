import pandas as pd
from sqlalchemy import create_engine

def get_engine():
    # --- Create a SQLAlchemy engine for database connection ---
    db_url="postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    return create_engine(db_url)

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
 

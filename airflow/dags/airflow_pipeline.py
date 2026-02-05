import os
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine, text

# --- Decoupling business logic from orchestration by importing from src ---
from src.db_utils import load_csv_to_db
from src.final_model import train_model
from src.feature_importance import get_feature_importance

# --- Metadata and retry policies for robust pipeline execution ---
default_args = {
    "owner": "airflow",
    "retries": 1
}

### --- Task Definitions (Modular Python Callables) ---

def task_load_csv():
    csv_path = os.getenv("CSV_PATH")
    load_csv_to_db(csv_path)

def task_run_etl():
    db_url = os.getenv("PROJECT_DB")
    engine = create_engine(db_url)

    sql_path = os.path.join(os.path.dirname(__file__), "initial_labeling.sql")
    with engine.begin() as connection:
        with open(sql_path, "r") as f:
            connection.execute(text(f.read()))

def task_train_model():
    train_model("/opt/airflow/models/pipeline_rf.joblib")

def task_feature_importance():
    get_feature_importance("/opt/airflow/models/pipeline_rf.joblib")


### --- DAG Configuration ---

with DAG(
    "airflow_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    # --- Step 1: Data Ingestion ---
    load_csv = PythonOperator(
        task_id="load_csv",
        python_callable=task_load_csv
    )

    # --- Step 2: Transformation & Labeling ---
    run_etl = PythonOperator(
        task_id="run_etl",
        python_callable=task_run_etl
    )

    # --- Step 3: Model Training ---
    train_model_ = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model
    )

    # --- Step 4: Model Interpretation ---
    feature_importance = PythonOperator(
        task_id="feature_importance",
        python_callable=task_feature_importance
    )

    # --- Task Dependencies (Pipeline Lineage) ---
    load_csv >> run_etl >> train_model_ >> feature_importance



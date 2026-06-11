from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Variables de entorno

sys.path.append("/opt/airflow")

def train_model_task():
    from src.ml_logic.train_mlflow import train
    # Importante: Dentro del contenedor, MLFLOW_TRACKING_URI debe ser el nombre del servicio
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    train()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "stroke_model_training",
    default_args=default_args,
    description="Entrena y registra el modelo de predicción de ACV en MLflow",
    schedule_interval=None, # Trigger manual por ahora
    catchup=False,
) as dag:

    training_step = PythonOperator(
        task_id="train_and_register_model",
        python_callable=train_model_task,
    )

    training_step

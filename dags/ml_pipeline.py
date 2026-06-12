from airflow.decorators import dag, task
from pendulum import datetime

import shutil
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from airflow.operators.bash import BashOperator 
from datetime import timedelta

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import data_loader


S3_OPTIONS = {
    "client_kwargs": {
        "endpoint_url": "http://s3:9000"
    }
}

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"


@dag(
    dag_id="stroke_mlops_pipeline",
    start_date=datetime(2026, 6, 1), 
    schedule=None,
    catchup=False,
    tags=["mlops", "tp_final"]
)
def tp_operaciones_maquina_pipeline():

    @task
    def tarea_1_preparar_datos():

        data_loader.split_and_store_data(
            input_path="s3://data/raw/healthcare-dataset-stroke-data.csv",
            train_path="s3://data/processed/train.parquet",
            test_path="s3://data/processed/test.parquet",
            storage_options=S3_OPTIONS
        )

        return "s3://data/processed/"

    @task
    def tarea_2_entrenar_modelo(data_prefix: str):

        train_df = pd.read_parquet(
            f"{data_prefix}train.parquet",
            storage_options=S3_OPTIONS
        )

        X_train = train_df.drop(columns=["stroke"])
        y_train = train_df["stroke"]

        cat_features = (
            X_train
            .select_dtypes(include=["object", "category"])
            .columns
            .tolist()
        )

        params = {
            "iterations": 150,
            "learning_rate": 0.05,
            "depth": 6,
            "auto_class_weights": "Balanced"
        }

        model = CatBoostClassifier(
            **params,
            verbose=False
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features
        )

        model_path = "/tmp/stroke_model.cbm"

        model.save_model(model_path)

        return {
            "model_path": model_path,
            "params": params,
            "data_prefix": data_prefix
        }

    @task
    def tarea_3_validar_modelo(training_info: dict):

        model_path = training_info["model_path"]
        data_prefix = training_info["data_prefix"]

        test_df = pd.read_parquet(
            f"{data_prefix}test.parquet",
            storage_options=S3_OPTIONS
        )

        X_test = test_df.drop(columns=["stroke"])
        y_test = test_df["stroke"]

        model = CatBoostClassifier()
        model.load_model(model_path)

        preds = model.predict(X_test)

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, preds)
            ),
            "precision": float(
                precision_score(y_test, preds)
            ),
            "recall": float(
                recall_score(y_test, preds)
            ),
            "f1": float(
                f1_score(y_test, preds)
            )
        }

        return {
            "model_path": model_path,
            "params": training_info["params"],
            "metrics": metrics
        }

    @task
    def tarea_4_registrar_modelo(validation_info: dict):

        # Ruta temporal limpia
        temp_dir = "/tmp/temp_model"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("Stroke_Prediction_Experiment")

        model_path = validation_info["model_path"]
        params = validation_info["params"]
        metrics = validation_info["metrics"]

        model = CatBoostClassifier()
        model.load_model(model_path)

        # 1. Iniciar el run para guardar métricas y artefactos
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            mlflow.catboost.save_model(cb_model=model, path="temp_model")
            mlflow.log_artifacts("temp_model", artifact_path="model")
            
            # Obtenemos la URI del artefacto manualmente
            model_uri = f"runs:/{run.info.run_id}/model"

        # 2. Registrar el modelo manualmente usando el Client
        client = MlflowClient()
        model_name = "Stroke_Classifier_Prod"
        
        try:
            client.create_registered_model(model_name)
        except:
            pass # Ya existe
        
        client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run.info.run_id
        )
        print("Modelo registrado con éxito mediante log_artifacts")

    notificar_api = BashOperator(
        task_id='notificar_api',
        bash_command='curl -X POST http://fastapi:8800/reload-model',
        retries=3,
        retry_delay=timedelta(seconds=10)
    )

    datos = tarea_1_preparar_datos()

    modelo = tarea_2_entrenar_modelo(datos)

    validacion = tarea_3_validar_modelo(modelo)

    registro = tarea_4_registrar_modelo(validacion)

    registro >> notificar_api

dag = tp_operaciones_maquina_pipeline()
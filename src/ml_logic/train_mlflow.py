import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Import utilities from our refactored src.ml_logic
try:
    from src.ml_logic.data_loader import load_dataset, get_train_test_split, clean_data_id_gender, impute_bmi_knn
    from src.ml_logic.evaluation import evaluate_model
except ImportError:
    from data_loader import load_dataset, get_train_test_split, clean_data_id_gender, impute_bmi_knn
    from evaluation import evaluate_model

# MLflow configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Stroke_Prediction_Experiment")

def train():
    with mlflow.start_run():
        # 1. Load and Clean
        df = load_dataset()
        df = clean_data_id_gender(df)
        
        # 2. Split
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        
        # 3. Impute BMI using KNN (as seen in teammate's logic)
        X_train, X_test = impute_bmi_knn(X_train, X_test)
        
        # 4. Define Pipeline
        categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        numerical_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
        
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )

        model_params = {
            "n_estimators": 100,
            "max_depth": None,
            "class_weight": "balanced",
            "random_state": 13
        }
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(**model_params))
        ])

        # 5. Log Parameters
        mlflow.log_params(model_params)
        
        # 6. Train
        print("Training Random Forest model...")
        pipeline.fit(X_train, y_train)
        
        # 7. Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: {metrics}")
        
        # 8. Log Model (Manual approach to avoid 404)
        print("Logging model manually as artifact...")
        temp_model_path = "temp_model_rf"
        import shutil
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)
            
        mlflow.sklearn.save_model(pipeline, temp_model_path)
        mlflow.log_artifacts(temp_model_path, artifact_path="model")
        print("Model artifacts logged in MLflow.")

if __name__ == "__main__":
    train()

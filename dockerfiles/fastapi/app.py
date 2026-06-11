from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="API de Predicción de ACV") # Titulo de la API

# Configuración de MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "Stroke_Prediction_Experiment"

mlflow.set_tracking_uri(MLFLOW_URI)

# Placeholder para el modelo cargado 
model = None

def load_model_from_mlflow():
    global model
    try:
        # Busca la última ejecución exitosa en el experimento
        print(f"Buscando la última ejecución en el experimento: {EXPERIMENT_NAME}")
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["start_time DESC"], max_results=1)
        
        if runs.empty:
            print(f"No se encontraron ejecuciones para el experimento {EXPERIMENT_NAME}")
            model = None
            return

        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"Intentando cargar el modelo desde: {model_uri}")
        
        # Carga el modelo directamente desde los artefactos
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Modelo cargado exitosamente desde la ejecución {run_id}.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None

# Evento que se ejecuta al iniciar la API
# 

@app.on_event("startup")
async def startup_event():
    load_model_from_mlflow()

@app.get("/")
def read_root():
    return {"message": "API de Predicción de ACV activa"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intente nuevamente.")
    
    try:
        # Convierte el diccionario de entrada a DataFrame
        df_input = pd.DataFrame([data])
        # El pipeline maneja el escalado y la codificación
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[:, 1]
        
        return {
            "prediction": int(prediction[0]),
            "stroke_probability": float(probability[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reload")
def reload_model():
    load_model_from_mlflow()
    return {"message": "Recarga del modelo activada", "success": model is not None}

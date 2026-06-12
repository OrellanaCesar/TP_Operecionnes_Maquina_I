from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import os
import time
from contextlib import asynccontextmanager

# Variables de entorno para que MLflow pueda descargar el modelo desde S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://s3:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# Configuración del Tracking URI de MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

# Variable global para mantener el modelo en memoria
model = None

# Definición del esquema de entrada (Features del dataset de Stroke)
class StrokeFeatures(BaseModel):
    gender: str = Field(..., description="Género del paciente (ej: 'Male', 'Female')")
    age: float = Field(..., gt=0, description="Edad del paciente en años")
    hypertension: int = Field(..., ge=0, le=1, description="¿Tiene hipertensión? (0=No, 1=Sí)")
    heart_disease: int = Field(..., ge=0, le=1, description="¿Tiene enfermedad cardíaca? (0=No, 1=Sí)")
    ever_married: str = Field(..., description="Estado civil ('Yes' o 'No')")
    work_type: str = Field(..., description="Tipo de ocupación del paciente")
    Residence_type: str = Field(..., description="Tipo de residencia ('Urban' o 'Rural')")
    avg_glucose_level: float = Field(..., gt=0, description="Nivel promedio de glucosa en sangre")
    bmi: float = Field(..., gt=0, description="Índice de masa corporal")
    smoking_status: str = Field(..., description="Estado de tabaquismo del paciente")

class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 67.0,
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked"
            }
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    retries = 5
    while retries > 0:
        try:
            print(f"Intentando cargar modelo... (intentos restantes: {retries})")
            model_uri = "models:/Stroke_Classifier_Prod/latest"
            model = mlflow.pyfunc.load_model(model_uri)
            print("Modelo cargado exitosamente.")
            break
        except Exception as e:
            print(f"Aún no disponible, esperando... ({e})")
            retries -= 1
            time.sleep(5) # Espera 5 segundos antes de reintentar
    yield
    print("Apagando API...")

app = FastAPI(
    title="Stroke Prediction MLOps API",
    description="API diseñada para predecir el riesgo de ACV basado en datos clínicos. Utiliza modelos registrados en MLflow.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post(
    "/predict", 
    summary="Realizar predicción de riesgo de ACV",
    description="Envía los datos clínicos de un paciente y recibe la predicción de riesgo (0 o 1).",
    responses={
        200: {"description": "Predicción exitosa"},
        503: {"description": "Modelo no disponible en memoria"}
    }
)
def predict(features: StrokeFeatures):
    """
    Toma los datos del paciente, los convierte a formato adecuado y retorna la clase predicha.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Revisa MLflow.")
    
    # Convertir el JSON de Pydantic a DataFrame de Pandas
    # El orient='index' y la transposición es para crear una fila (1, N)
    df = pd.DataFrame.from_dict(features.model_dump(), orient='index').T
    
    try:
        prediction = model.predict(df)
        # prediction[0] suele devolver un numpy.int64, lo casteamos a int normal
        return {
            "stroke_risk_prediction": int(prediction[0]),
            "model_version": "latest"
        }
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Error en inferencia: {str(e)}")


@app.post("/reload-model", summary="Recargar modelo desde MLflow", description="Úsalo después de entrenar para actualizar el modelo en memoria.")
async def reload_model():
    global model
    try:
        model = mlflow.pyfunc.load_model("models:/Stroke_Classifier_Prod/latest")
        return {"message": "Modelo recargado exitosamente", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar: {str(e)}")
        

@app.get(
    "/health", 
    summary="Verificar estado del servicio",
    description="Retorna el estado de salud de la API y si el modelo está cargado en memoria."
)
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
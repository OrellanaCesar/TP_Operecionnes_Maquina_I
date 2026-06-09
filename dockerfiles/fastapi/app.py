from fastapi import FastAPI
import mlflow
import os

app = FastAPI(title="MLOps Model Serving API")

@app.get("/")
def read_root():
    return {"message": "API de Servido de Modelos MLOps activa"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Trabajo Práctico: Operaciones de Máquina I

## MLOps - CEIA - FIUBA

### Integrantes
* **Britez, Leandro Adrian**
* **Orellana, César Andrés**

---

## 🚀 Descripción del Proyecto
Este repositorio implementa una plataforma de **MLOps automatizada y end-to-end**. El sistema gestiona todo el ciclo de vida de un modelo de predicción de riesgo de ACV (*Stroke Prediction*), garantizando trazabilidad, reproducibilidad y persistencia de datos y modelos.

## 🧠 El Modelo
Se utiliza un **CatBoostClassifier** para la predicción de ACV. Este algoritmo fue seleccionado por su excelente manejo nativo de variables categóricas (como tipo de trabajo o estado de fumador) y su robustez frente al desbalance de clases típico en datos médicos.

## 🏗️ Arquitectura del Sistema
El entorno está orquestado mediante **Docker Compose (11 servicios)** e incluye:

*   **Orquestación (Apache Airflow 3)**: Gestión de pipelines para la preparación de datos, entrenamiento, validación y registro automático de modelos. Pipeline con 5 tareas usando `CeleryExecutor` y `Redis`.
*   **Gestión de Experimentos (MLflow)**: Tracking de métricas (Accuracy, Recall, F1) y **Model Registry**.
*   **Data Lake (MinIO - S3)**: Almacenamiento de datos crudos, procesados y artefactos pesados.
*   **Base de Datos (PostgreSQL)**: Persistencia de metadatos de Airflow y MLflow.
*   **Serving (FastAPI)**: API de inferencia con carga dinámica y persistencia entre reinicios.

---

## 🛠️ Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone [url-del-repo]
cd TP_Operaciones_Maquina_I
```

### 2. Levantar los servicios

```bash
docker compose --profile all up -d --build
```

Durante el primer arranque, el contenedor `create_s3_buckets` inicializa automáticamente el almacenamiento MinIO, crea los buckets necesarios y carga el dataset crudo (`healthcare-dataset-stroke-data.csv`) en el Data Lake.

### 3. Verificar el estado de los servicios

```bash
docker compose ps
```

Para revisar los logs de inicialización:

```bash
docker compose logs create_s3_buckets
docker compose logs airflow-init --tail 50
```

### 4. Acceso a los servicios

| Servicio | URL |
|-----------|------|
| Airflow UI | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |
| MinIO Console | http://localhost:9001 |
| FastAPI Swagger | http://localhost:8800/docs |

**Credenciales MinIO**

```text
Usuario: minio
Password: minio123
```

---

## 🚀 Funcionamiento del Pipeline

El sistema opera de forma integrada para automatizar el ciclo completo de vida del modelo.

### Flujo de Inicialización

Al iniciar la plataforma:

1. Se levanta PostgreSQL para almacenar metadatos.
2. Se inicia MinIO como almacenamiento compatible con S3.
3. El contenedor `create_s3_buckets` crea los buckets y carga el dataset inicial.
4. `airflow-init` inicializa la base de datos de Airflow.
5. Se levantan Redis y los componentes de Airflow.
6. Se inicia MLflow utilizando PostgreSQL y MinIO.
7. Finalmente se inicia FastAPI, que utilizará los modelos registrados en MLflow para realizar inferencias.

### Pipeline de Entrenamiento

Al ejecutar el DAG `stroke_mlops_pipeline`:

1. **Ingestion**
   - Se leen los datos desde `s3://data/raw`.

2. **Preprocessing**
   - Se eliminan columnas irrelevantes.
   - Se corrigen valores inconsistentes.
   - Se imputan valores faltantes utilizando KNN Imputer.
   - El dataset procesado se almacena en `s3://data/processed`.

3. **Training**
   - Se entrena un modelo CatBoost para predicción de riesgo de ACV.
   - Se utilizan pesos automáticos para manejar el desbalanceo de clases.

4. **Model Registry**
   - El modelo entrenado se registra automáticamente en MLflow como `Stroke_Classifier_Prod`.
   - Se almacenan métricas, parámetros y artefactos asociados al experimento.

5. **Serving Hot-Reload**
   - Airflow ejecuta una tarea que invoca el endpoint `POST /reload-model`.
   - FastAPI descarga la última versión registrada desde MLflow y actualiza el modelo en memoria sin necesidad de reiniciar el servicio.

---

## 🧪 Guía de Pruebas

### Ejecutar el pipeline

1. Acceder a Airflow:
   - http://localhost:8080

2. Buscar el DAG:

```text
stroke_mlops_pipeline
```

3. Ejecutar:

```text
Trigger DAG
```

### Monitorear la ejecución

- Verificar que las tareas del DAG finalicen correctamente.
- Confirmar que la tarea de notificación a la API complete exitosamente.

### Verificar el modelo cargado

Acceder a:

```text
http://localhost:8800/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Realizar inferencias

1. Abrir Swagger:
   - http://localhost:8800/docs

2. Ejecutar el endpoint:

```text
POST /predict
```

3. Ingresar los datos del paciente y obtener la predicción en tiempo real.

---

## 💾 Persistencia de Datos

La plataforma utiliza los volúmenes Docker:

```text
db_data
minio_data
```

Esto garantiza que:

- Los experimentos y registros de MLflow se mantengan entre reinicios.
- Los modelos almacenados en MinIO no se pierdan.
- Los datasets procesados continúen disponibles.
- La API pueda cargar automáticamente la última versión registrada del modelo al reiniciar la plataforma.

---

## 📂 Estructura del Repositorio

```text
├── airflow/            # Configuraciones y DAGs de Airflow
│   └── dags/           # Flujos de trabajo automatizados
├── data/               # Dataset original para carga inicial
├── dockerfiles/        # Imágenes Docker de los servicios
├── fastapi/            # API de inferencia
├── training/           # Entrenamiento y validación del modelo
├── mlruns/             # Experimentos y artefactos generados por MLflow
├── docker-compose.yml  # Orquestación de la infraestructura
└── README.md           # Documentación del proyecto
```
# Trabajo Práctico: Operaciones de Máquina I

## MLOps - CEIA - FIUBA

### Integrantes
* **Britez, Leandro Adrian**
* **Orellana, César Andrés**

---

## Descripción del Proyecto
Este repositorio implementa una plataforma de **MLOps automatizada y end-to-end**. El sistema gestiona todo el ciclo de vida de un modelo de predicción de riesgo de ACV (*Stroke Prediction*), desde la ingesta de datos hasta el despliegue en producción, garantizando trazabilidad y reproducibilidad.

## Arquitectura del Sistema
El entorno está orquestado mediante contenedores Docker e incluye los siguientes componentes:

* **Orquestación (Apache Airflow)**: Gestión de pipelines para la preparación de datos, entrenamiento, validación y registro automático de modelos.
* **Gestión de Experimentos y Modelos (MLflow)**: Tracking de métricas y Registro de Modelos (*Model Registry*).
* **Almacenamiento (MinIO & PostgreSQL)**: Almacenamiento de objetos S3 para artefactos y base de datos relacional para metadatos.
* **Serving (FastAPI)**: API de inferencia con documentación automática (Swagger) y carga dinámica de modelos desde el Registry.



## Instalación y Uso

1. **Clonar el repositorio**:
   ```bash
   git clone [url-del-repo]

2. **Levantar los servicios**:
```bash
docker compose up -d --build
```

3. **Acceso a los servicios**:
   * **MLflow UI**: [http://localhost:5000](http://localhost:5000)
   * **MinIO Console**: [http://localhost:9001](http://localhost:9001)
   * **FastAPI (Swagger)**: [http://localhost:8800/docs](http://localhost:8800/docs)
   * **Airflow UI**: [http://localhost:8080](http://localhost:8080) 

4. Funcionamiento del Pipeline

El sistema opera de forma integrada para minimizar la intervención manual:

* **Pipeline de Entrenamiento**: Al disparar el DAG en Airflow, se ejecuta secuencialmente la preparación de datos, el entrenamiento con CatBoost y la validación de métricas.
* **Registro Automático**: El modelo es registrado en el Model Registry de MLflow como Stroke_Classifier_Prod.
* **Hot-Reloading**: La tarea_5 (BashOperator) envía una señal a la API mediante una petición POST al endpoint /reload-model. Esto fuerza a la API a recargar el modelo desde el Registry sin necesidad de reiniciar el contenedor.

5. Guía de Pruebas (Paso a Paso)

* **Ejecutar el pipeline**:
    * Dirígete a http://localhost:8080.
    * Busca el DAG stroke_mlops_pipeline.
    * Haz clic en Trigger DAG (botón azul).
* **Monitorear**: 
    * Observa cómo las 5 tareas se ejecutan y cambian a color verde en el grafo de Airflow.
    * La notificar_api confirmará que la API recibió el modelo nuevo.
* **Verificar el modelo**:
    * Entra a http://localhost:8800/health.
    * Deberías ver: {"status": "ok", "model_loaded": true}.
* **Realizar Inferencia**:
    * Ve a http://localhost:8800/docs.
    * Haz clic en el endpoint POST /predict.
    * Presiona "Try it out", rellena los datos del paciente y haz clic en Execute para ver la predicción en tiempo real.


## Estructura del Repositorio
```text
├── airflow/            # Configuraciones y DAGs
│   └── dags/           # Flujos de trabajo automatizados
├── dockerfiles/        # Definiciones de imágenes Docker
├── fastapi/            # Código de la API de inferencia
├── training/           # Scripts de entrenamiento y validación
├── mlruns/             # (Generated) Logs de experimentos de MLflow
├── docker-compose.yml  # Orquestación de servicios
└── README.md           # Documentación del proyecto
```

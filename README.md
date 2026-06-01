# Trabajo Práctico: Operaciones de Máquina I

## MLOps - CEIA - FIUBA

### Integrantes
* **Britez, Leandro Adrian**
* **Orellana, César Andrés**

---

## Descripción del Proyecto
Este repositorio contiene la implementación de un entorno productivo para modelos de Machine Learning, siguiendo los principios de MLOps. La infraestructura está diseñada para soportar tareas de **DataOps** y **MLOps** mediante el uso de contenedores Docker.

## Estado de la Primera Entrega
Se ha establecido la infraestructura base necesaria para la gestión del ciclo de vida de los modelos:

* **Gestión de Experimentos (MLflow)**: Servidor centralizado para el seguimiento de métricas, parámetros y versiones de modelos.
* **Almacenamiento Estructurado (PostgreSQL)**: Backend Store para la persistencia de metadatos de MLflow.
* **Almacenamiento de Artefactos (MinIO/S3)**: Artifact Store local compatible con S3 para guardar modelos y datasets de forma escalable.

*(Nota: Próximamente se integrarán servicios de orquestación (Airflow) y servicio de modelos (FastAPI))*

## Requisitos Previos
* [Docker](https://www.docker.com/get-started) y Docker Compose instalados.
* [uv](https://github.com/astral-sh/uv) (opcional, para gestión local de dependencias).

## Instalación y Uso

1. **Clonar el repositorio**:
   ```bash
   git clone [url-del-repo]
   ```

2. **Levantar los servicios**:
   ```bash
   docker compose up -d
   ```

3. **Acceso a los servicios**:
   * **MLflow UI**: [http://localhost:5000](http://localhost:5000)
   * **MinIO Console**: [http://localhost:9001](http://localhost:9001)

## Estructura del Repositorio
```text
├── airflow/            # Configuraciones y DAGs (en desarrollo)
├── dockerfiles/        # Definiciones de imágenes personalizadas
│   ├── mlflow/
│   └── postgresql/
├── main.py             # Script principal de prueba
├── docker-compose.yml  # Orquestación de contenedores
└── pyproject.toml      # Configuración de dependencias (uv)
```
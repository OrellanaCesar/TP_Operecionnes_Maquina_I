# Usa una imagen base de Python pequeña (versión 3.10) optimizada para producción
FROM python:3.10-slim

# Copia el archivo de requisitos de MLflow al contenedor
COPY requirements.txt /requirements.txt

# Instala las librerías necesarias sin guardar caché para mantener la imagen ligera
RUN pip install --no-cache-dir -r /requirements.txt

# Expone el puerto 5000, que es donde MLflow escuchará las conexiones
EXPOSE 5000

# Define el comando inicial: al arrancar el contenedor, ejecutará el servidor de MLflow
ENTRYPOINT ["mlflow", "server"]
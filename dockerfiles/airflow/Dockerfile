# Usa la imagen base oficial de Apache Airflow, versión 3.0.2
FROM apache/airflow:3.0.2

# Copia tu archivo de librerías desde tu PC al interior del contenedor
COPY requirements.txt /requirements.txt

# Cambia al usuario 'root' para tener permisos de administrador y realizar instalaciones del sistema
USER root

# Actualiza los repositorios de paquetes e instala 'git' (útil para bajar código de repositorios)
# Luego limpia los archivos temporales para reducir el tamaño de la imagen
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && apt-get autoremove -yqq --purge \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Vuelve al usuario 'airflow' por seguridad (nunca debemos ejecutar procesos como root si no es necesario)
USER airflow

# Providers necesarios para Celery + Redis
RUN pip install --no-cache-dir \
    apache-airflow-providers-celery \
    redis
    
# Instala todas las librerías de Python listadas en el requirements.txt
# --no-cache-dir evita que se guarden archivos temporales de instalación (ahorra espacio)
RUN pip install --no-cache-dir -r /requirements.txt
# Usa    la imagen oficial de python 3.10 slim
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el  resto de los archivos al directorio de trabajo
COPY . .

# Expone el puerto 8800
EXPOSE 8800

# Comando que se ejecuta al iniciar el contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]

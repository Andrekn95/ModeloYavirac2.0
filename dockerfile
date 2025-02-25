FROM python:3.9-slim

# Instalar dependencias del sistema para entrenamiento (ej. CUDA si usas GPU)
RUN apt-get update && apt-get install -y \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar TODO el proyecto (excepto lo excluido en .dockerignore)
COPY . .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando por defecto (ajustar según necesidad)
CMD ["python", "app.py"]  # Para inferencia

# Opcional: Si quieres permitir entrenamiento
# ENTRYPOINT ["python", "train.py"]
# 1. Image Python officielle légère (Linux Debian slim)
FROM python:3.12-slim

# 2. Dossier de travail et configuration du PYTHONPATH
WORKDIR /app
ENV PYTHONPATH=/app

# 3. Installer libgomp1 (requis par XGBoost sous Linux)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le code source, les tests et les modèles
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/

# 6. Exposer le port 8000
EXPOSE 8000

# 7. Lancer le serveur FastAPI au démarrage du conteneur
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
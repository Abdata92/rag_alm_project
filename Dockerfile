# Image Python légère de base
FROM python:3.11-slim

# Éviter la création de fichiers .pyc et forcer l'affichage immédiat des logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN pip install --no-cache-dir poetry

# Copie des fichiers de dépendances
COPY pyproject.toml poetry.lock /app/

# Configuration de Poetry pour installer les packages directement dans le conteneur
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copie du reste du projet
COPY . /app

# Exposition du port Streamlit
EXPOSE 8501

# Lancement de l'interface Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
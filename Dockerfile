# Image de base
FROM python:3.10-slim

# Créer le dossier de travail
WORKDIR /app

# Copier les fichiers dans l’image
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Uvicorn
EXPOSE 8000

# Commande pour lancer FastAPI via uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
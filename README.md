# XOR FastAPI API

Une API REST construite avec FastAPI pour prédire le résultat de l'opération logique XOR (exclusive OR) à l'aide d'un réseau de neurones simple entraîné avec PyTorch.

---

## Fonctionnalités

- Modèle PyTorch entraîné sur les 4 cas du XOR
- Prédiction via une API `/predict`
- Dockerisé pour être lancé partout facilement
- Retour JSON propre : entrée, probabilité (raw_output), classe prédite

---

## Structure du projet

xor-fastapi-api/
├── model.py              → Architecture du modèle
├── train.py              → Entraînement du modèle
├── main.py               → API FastAPI
├── test.py               → Script de test local
├── saved_model/
│   └── xor_model.pth     → Modèle sauvegardé
├── requirements.txt
├── Dockerfile
└── README.md

---

## Entraîner le modèle (optionnel)

Commande à exécuter :

    python train.py

Cela entraîne le modèle XOR et sauvegarde le fichier xor_model.pth dans saved_model/.

---

## Lancer l’API localement (sans Docker)

1. Installer les dépendances :

    pip install -r requirements.txt

2. Lancer le serveur FastAPI :

    uvicorn main:app --reload

Accès à la documentation interactive :  
http://localhost:8000/docs

---

## Tester l’API avec curl

    curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"x1": 1, "x2": 0}'

Réponse attendue :

    {
      "input": [1, 0],
      "raw_output": 0.9998,
      "prediction": 1
    }

---

## Lancer avec Docker

1. Build de l’image :

    docker build -t xor-api .

2. Lancer le conteneur :

    docker run -p 8000:8000 --name xor-api xor-api

Accès à l’API :  
http://localhost:8000/docs

---

## Technologies utilisées

- Python 3.10
- FastAPI
- PyTorch
- Docker
- Uvicorn
# 🔌 XOR FastAPI API

A simple REST API built with **FastAPI** to predict the result of the logical XOR operation using a neural network model trained with **PyTorch**.

---

## 🚀 Features

- 🧠 PyTorch model trained on the 4 basic XOR cases
- 🔁 Prediction via a `/predict` POST endpoint
- 🐳 Fully dockerized for easy deployment
- 📊 Clean JSON response: input, raw_output (probability), final class

---

## ⚙️ Train the model (optional)

Run the training script:

    python train.py

This will train the XOR neural network and save the weights to `saved_model/xor_model.pth`.

---

## 🚀 Run the API locally (without Docker)

1. Install dependencies:

    pip install -r requirements.txt

2. Start the FastAPI server:

    uvicorn main:app --reload

API documentation available at:  
http://localhost:8000/docs

---

## 📬 Test the API with curl

    curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"x1": 1, "x2": 0}'

Expected response:

    {
      "input": [1, 0],
      "raw_output": 0.9998,
      "prediction": 1
    }

---

## 🐳 Run with Docker

1. Build the image:

    docker build -t xor-api .

2. Run the container:

    docker run -p 8000:8000 --name xor-api xor-api

Access the API at:  
http://localhost:8000/docs

---

## 📌 Technologies Used

- Python 3.10  
- FastAPI  
- PyTorch  
- Docker  
- Uvicorn

The trained model (`.pth` file) is included and was trained entirely from scratch on synthetic data.

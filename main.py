from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import XORNet

app = FastAPI()
model = XORNet()
model.load_state_dict(torch.load("saved_model/xor_model.pth"))
model.eval()

class InputData(BaseModel):
    x1: float
    x2: float

@app.post("/predict")
def predict(data: InputData):
    inputs = torch.tensor([[data.x1, data.x2]])
    with torch.no_grad():
        output = model(inputs)
        prediction = int((output > 0.5).item())
    return {
        "input": [data.x1, data.x2],
        "raw_output": round(output.item(), 4),
        "prediction": prediction
    }

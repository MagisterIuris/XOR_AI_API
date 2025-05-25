import torch
import torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return x

model = XORNet()
model.load_state_dict(torch.load("saved_model/xor_model.pth"))
model.eval()

test_data = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0.99, 0.01],
    [0.01, 0.99],
    [0.5, 0.5],
    [0.9, 0.9],
    [1.2, 0.0],
    [0.0, 1.2]
], dtype=torch.float32)

with torch.no_grad():
    predictions = model(test_data)
    predictions_bin = (predictions > 0.5).float()

print("🧪 Prédictions du modèle XOR :")
for i in range(len(test_data)):
    print(f"Entrée : {test_data[i].tolist()} → Prédit : {predictions[i].item():.4f} → Classe : {int(predictions_bin[i].item())}")

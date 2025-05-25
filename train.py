import torch 
from model import XORNet
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt

entrees= torch.tensor ([
    [0, 0], 
    [0, 1], 
    [1, 0],  
    [1, 1]  
], dtype=torch.float32)

sorties = torch.tensor([
    [0],
    [1],
    [1],
    [0],
], dtype=torch.float32)


XOR_model = XORNet() 

critere = nn.BCELoss() 
optimizer = optim.SGD(XOR_model.parameters(), lr= 0.4) 

loss_history = []
for epoch in range (100000): 
    output = XOR_model(entrees) 
    print("output :", output)
    loss = critere(output, sorties)
    loss_history.append(loss.item())

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if epoch % 1000 == 0:
        print(f"Époque {epoch} - Perte : {loss.item():.4f}")

plt.plot(loss_history)
plt.title("Courbe de perte (loss) pendant l'entraînement")
plt.xlabel("Itération (epoch)")
plt.ylabel("Loss (BCE)")
plt.grid(True)
plt.show()


torch.save(XOR_model.state_dict(), "saved_model/xor_model.pth")
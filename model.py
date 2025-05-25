import torch.nn as nn

class XORNet(nn.Module): 
    def __init__(self): 
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, entree): 
        entree = self.activation(self.layer1(entree))              
        entree = self.activation(self.layer2(entree))
        return entree
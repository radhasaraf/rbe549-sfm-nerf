import torch
import torch.nn as nn

class NeRF(nn.Module):
    """
    input layer will have 3 inputs 
    last but one layer will have additional 3 inputs
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU
        self.lin1 = nn.Linear(3, 128) # inputs are x, y, z
        self.lin2 = nn.Linear(128,128)
        self.lin3 = nn.Linear(128,128)
        self.lin4 = nn.Linear(128,64)  # assuming that we are not using viewing directions for now
        self.lin5 = nn.Linear(64, 4)  # outputs are RGB, sigma

    def forward(self, x):  # (H*W) x 3
        print(x.get_device())
        x = self.lin1(x)
        x = self.relu(x)

        x = self.lin2(x)
        x = self.relu(x)

        x = self.lin3(x)
        x = self.relu(x)

        x = self.lin4(x)
        x = self.relu(x)

        x = self.lin5(x)
        x = self.relu(x)

        return x  # H*W x 4

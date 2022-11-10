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
        self.lin1 = nn.Sequential(
                            nn.Linear(24,128),
                            nn.ReLU()
                        )# input is x, y, z not using positional encoding now
        self.lin2 = nn.Sequential(
                            nn.Linear(128,128),
                            nn.ReLU()
                        )
        self.lin3 = nn.Sequential(
                            nn.Linear(128,128),
                            nn.ReLU()
                        )
        self.lin4 = nn.Sequential(
                            nn.Linear(128,64),
                            nn.ReLU()
                        )# assuming that we are not using viewing directions for now
        self.lin5 = nn.Sequential(
                            nn.Linear(64,4),
                            nn.ReLU()
                        )# outputs are RGB, sigma

    def forward(self, x):  # (H*W) x 3
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        return x  # H*W x 4

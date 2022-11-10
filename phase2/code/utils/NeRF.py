import torch
import torch.nn as nn

class NeRF(nn.Module):
    """
    input layer will have 3 inputs 
    last but one layer will have additional 3 inputs
    """
    def __init__(self, input_channels):
        super().__init__()
        self.lin1 = nn.Sequential(nn.Linear(input_channels,256), nn.ReLU())
        self.lin2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.lin3 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.lin4 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.lin5 = nn.Sequential(nn.Linear(256 + input_channels, 256), nn.ReLU())
        self.lin6 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.lin7 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.lin8 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.lin9 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.volume_density = nn.Sequential(nn.Linear(256,1), nn.ReLU())

        self.lin10 = nn.Sequential(nn.Linear(256,128), nn.ReLU())
        self.lin11 = nn.Sequential(nn.Linear(128,3), nn.Sigmoid())

    def forward(self, x):  # (H*W*n_samples) x 3
        residual = x
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(torch.cat([x , residual], axis=-1))
        x = self.lin6(x)
        x = self.lin7(x)
        x = self.lin8(x)
        x = self.lin9(x)
        sigma = self.volume_density(x)
        x = self.lin10(x)
        rgb = self.lin11(x)
        rgbs = torch.cat([rgb, sigma], axis=-1)
        return rgbs  # (H*W*n_samples) x 4

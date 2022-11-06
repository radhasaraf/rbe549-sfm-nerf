import numpy as np

def load_tiny_nerf():
    data = np.load("../../data/tiny_nerf_data.npz")
    return data


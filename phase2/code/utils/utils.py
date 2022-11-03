import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from skimage import io

def load_tiny_nerf():
    data = np.load("../../data/tiny_nerf_data.npz")
    return data

class NeRFDatasetLoader(Dataset):
    """
    NeRF Dataset loader class
    tutorial - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, dataset_path, mode):
        """
        dataset_path - path to the dataset, 
            assumes the following folder structure
            .
            ├── test
            ├── train
            │   ├── r_0.png
            │   ├── r_1.png
            │   ├── r_10.png
            ├── transforms_test.json
            ├── transforms_train.json
            ├── transforms_val.json
            └── val
        mode - train/test/val
        """
        self.root_dir = dataset_path
        if mode == "test":
            file = "transforms_test.json"
        elif mode == "train":
            file = "transforms_train.json"
        elif mode == "val":
            file = "transforms_val.json"

        transforms_path = os.path.join(dataset_path + os.sep + file)
        print(f"Loading {transforms_path} for {mode}")
        with open(transforms_path) as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data["frames"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_file = self.data["frames"][idx]["file_path"] + ".png"
        img_name = os.path.join(self.root_dir + os.sep + img_file)
        image = io.imread(img_name)
        transforms = self.data["frames"][idx]["transform_matrix"]
        transforms = torch.tensor(transforms)
        return {'image': image, 'transforms': transforms}
    
def load_lego_data():
    """
    inputs:
        path - lego dataset path
    outputs:
        
    """
    pass

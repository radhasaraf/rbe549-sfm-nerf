import numpy as np
import argparse
import json
from utils.utils import NeRFDatasetLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main(args):
    lego_dataset = NeRFDatasetLoader(args.datasetPath, "val")
    dataloader = DataLoader(lego_dataset, batch_size = 4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['transforms'].size())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")
    parser.add_argument('--datasetPath',default='../data/lego/',help="dataset path")

    args = parser.parse_args()
    main(args)


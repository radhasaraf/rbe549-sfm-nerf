import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import argparse
import json
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

from utils.NeRFDatasetLoader import NeRFDatasetLoader

def main(args):
    lego_dataset = NeRFDatasetLoader(args.datasetPath, "val")
    data  = lego_dataset[2]
    print(data["focal_length"]) #TODO need to verify
    #dataloader = DataLoader(lego_dataset, batch_size = 4, shuffle=True, num_workers=0)

#    for i_batch, sample_batched in enumerate(dataloader):
#        print(i_batch, sample_batched['image'].size(), sample_batched['transforms'].size())
    # loop over the camera views
    # for each view get the estimated RGB image
    # if it is in Train:
    #   compare the photometric loss
    #   backpropagate
    # if it is not in Train:
    #   get the RGB image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")
    parser.add_argument('--datasetPath',default='../data/lego/',help="dataset path")

    args = parser.parse_args()
    main(args)


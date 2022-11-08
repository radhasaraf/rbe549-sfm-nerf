import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import argparse
import json
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.NeRFDatasetLoader import NeRFDatasetLoader

def get_rays(H, W, f, R, T):
    """
    r = o + t*d (o,d)
    inputs:
        H - 1, - height
        W - 1, - width 
        f - 1, - focal length
        R - 3 x 3 - camera rotation matrix wrt camera
        T - 3 x 3 - camera translation vector wrt camera
    outputs:
        ray_directions - H x W x 3
        ray_origins - H x W x 3
    """
    # create meshgrid
    xs = torch.linspace(0, W-1, W)  # 1 x W
    ys = torch.linspace(0, H-1, H)  # 1 x H
    x, y = torch.meshgrid(xs, ys)   # W x H, W x H
    x = x.t()  # H x W
    y = y.t()  # H x W

    x = (x - W/2)/f  # H x W
    y = (y - H/2)/f  # H x W
    # get directions of each pixel assuming camera at origin
    ray_directions = torch.stack([x, -y, torch.ones_like(x)], axis=-1)  # H x W x 3
    # TODO
    # H*W x 3
    # 3 x H*W
    # R (3 x 3) * (3 x H*W)
    # 3 x H*W
    # (H*W) x 3
    # H x W x 3

    # apply rotation to the directions
    ray_directions = ray_directions[..., None, :]  # H x W x 1 x 3
    ray_directions = ray_directions*R  # H x W x 3 x 3 = H x W x 1 x 3 * 3 x 3
    ray_directions = torch.sum(ray_directions, -1) # H x W x 3
    ray_origins = torch.broadcast_to(T, torch.shape(ray_directions)) # H x W x 3
    return ray_directions, ray_origins

def generate_ray_points():
    """
    inputs:
        ray_origins 
        ray_directions
        N_samples
        N_rand_rays - None
        t_near
        t_far
    outputs
        ray_points - 
    """
    # TODO need to consider N_rand_rays case
    pass

def main(args):
    lego_dataset = NeRFDatasetLoader(args.datasetPath, "val")
    data  = lego_dataset[2]
    print(data["focal_length"]) #TODO need to verify
    #dataloader = DataLoader(lego_dataset, batch_size = 4, shuffle=True, num_workers=0)

    # loop over the camera views
    # for each view get the estimated RGB image
    # if it is in Train:
    #   compare the photometric loss
    #   backpropagate
    # if it is not in Train:
    #   get the RGB image

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")
    parser.add_argument('--datasetPath',default='../data/lego/',help="dataset path")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)


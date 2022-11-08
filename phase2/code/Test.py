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
    r = o + t*d 
    Building o and d here 
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

def generate_ray_points(ray_directions, ray_origins, N_samples, t_near=0, t_far=1, N_rand_rays=None):
    """
    r = o + t*d 
    building r here
    inputs:
        ray_directions - H x W x 3
        ray_origins  - H x W x 3
        N_samples
        t_near
        t_far
        N_rand_rays - None
    outputs
        ray_points - H*W x 3
    """
    # TODO need to consider N_rand_rays case
    ts = torch.linspace(t_near, t_far, N_samples)  # N, 
    rays = ray_origins[..., None, :] + ray_directions[..., None, :]*ts[..., None]  # H x W x 1 x 3 #TODO how is this working? 
    points = rays.reshape((-1,3))
    #points = encode_position(points)
    return points, ts

def test(args):
    pass

def train(args):
    lego_dataset = NeRFDatasetLoader(args.path)
    f, transforms, images = lego_dataset.get_full_data()
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))


    model.train()
    for i_iter in range(args.max_iters):
        for transform, gt_image in zip(transforms, images):
            H, W = gt_image.shape
            R = transform[:3,:3]
            T = transform[:,3]
            ray_dirs, ray_origins = get_rays(H, W, f, R, T)  # H x W x 3
            ray_points, ts = generate_ray_points(ray_dirs, ray_origins, args.n_ray_points)  # H x W x 3
            rgbs = model(ray_points)  # H*W x 4
            rgb = rgbs[:,:3].reshape((H, W))  # H x W x 3
            s = rgbs[:,3]

            train_image = volumetric_rendering(rgb, s)
            loss_this_iter = loss(gt_image, train_image)

            optimizer.zero_grad()
            loss_this_iter.backward()
            optimizer.step()
        
            # NOTE: from pytorch implementation of NeRF
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate



def main(args):

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


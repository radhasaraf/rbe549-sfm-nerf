import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import argparse
import json
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.NeRFDatasetLoader import NeRFDatasetLoader
from utils.NeRF import NeRF
import cv2
from tqdm import tqdm
import random

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    return device

device = get_device()
print(f"Running on device: {device}")

def encode_positions(x, n_dim=4):
    """
    Encodes positions into higher dimension
    inputs:
        x - (H*W*n_samples) x 3
        n_dim - 1, - number of dimensions it has to encode
    outpus:
        y - (H*W*n_samples) x (3*n_dim)
    """
    positions = []
    #TODO it's crashing here if the image size is 800 x 800
    for i in range(n_dim):
        for fn in [torch.sin, torch.cos]:
            positions.append(fn((2.0**i)*x))
    return torch.concat(positions, axis=-1)

def volumetric_rendering(raw, ts, ray_directions):
    """
    takes raw outputs from network and returns image
    inputs:
        raw - n_rays(H*W) x n_ray_points x 4
        ts - N, -  parameter t along z axis 
        ray_directions - H x W x 3
    outputs:
        image -  H x W viewed along that direction
    """
    # rgb = torch.sigmoid(raw[..., :3])  # n_rays x n_ray_points x 3
    rgb = raw[..., :3]  # n_rays x n_ray_points x 3

    delta_ts = ts[..., 1:] - ts[..., :-1] # n_ray_points-1,
    t1 = torch.Tensor([1e10]).expand(delta_ts[...,:1].shape).to(device)
    delta_ts = torch.cat([delta_ts, t1], -1)  # [n_ray_points,]


    ray_directions = ray_directions.reshape(rgb.shape[0], -1)  # n_rays(H*W) x 3
    ray_directions_norm = torch.norm(ray_directions[..., None, :], dim =-1) # n_rays x 1 x 3
    delta_ts = delta_ts * ray_directions_norm  # n_rays x n_ray_points x 3

    raw2alpha = lambda raw, delta_ts: 1.-torch.exp(-raw*delta_ts)
    alpha = raw2alpha(raw[...,3], delta_ts)  # [N_rays, N_samples]
    t1 = torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1)
    weights = alpha * torch.cumprod(t1, -1)[:, :-1]
    weights = weights.to(device)
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    return rgb_map

def get_rays(image, f, R, T):
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
    xs = torch.linspace(0, image.shape[0]-1, image.shape[0]).to(device)  # 1 x W
    ys = torch.linspace(0, image.shape[1]-1, image.shape[1]).to(device)  # 1 x H
    x, y = torch.meshgrid(xs, ys)   # W x H, W x H
    x = x.t()  # H x W
    y = y.t()  # H x W

    x = (x - image.shape[0]/2)/f  # H x W
    y = (y - image.shape[1]/2)/f  # H x W
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
    ray_origins = torch.broadcast_to(T, ray_directions.shape) # H x W x 3
    return ray_directions, ray_origins

def generate_ray_points(ray_directions, ray_origins, n_ray_point_samples, t_near=0, t_far=1, N_rand_rays=None, n_frequencies=4):
    """
    r = o + t*d 
    building r here
    inputs:
        ray_directions - (n_rays) x 3
        ray_origins  - (n_rays) x 3
        n_ray_point_samples
        t_near
        t_far
        N_rand_rays - None
    outputs
        ray_points - H*W*n_samples x 3
        ts - samples
    """
    ts = torch.linspace(t_near, t_far, n_ray_point_samples).to(device)  # N, 
    rays = ray_origins[..., None, :] + ray_directions[..., None, :]*ts[..., None]  # (n_rays x n_samples x 3 
    points = rays.reshape((-1,3))  # (n_rays*n_samples) x 3
    points = encode_positions(points, n_frequencies)  # (n_rays*n_samples) x (3*4)
    return points, ts

def test(args):
    pass

def photometric_loss(gt_image, train_image):
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    img_loss = torch.mean((train_image - gt_image)**2)
    return img_loss

def train(args):
    lego_dataset = NeRFDatasetLoader(args.dataset_path, args.mode)
    f, transforms, images = lego_dataset.get_full_data(device)

    # get input channels
    height, width, _ = images[0].shape
    input_channels = 3*2*args.n_pose_frequencies

    # initialize the model
    model = NeRF(input_channels).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    model.train()
    for i_iter in tqdm(range(args.max_iters)):
        for i in tqdm(range(transforms.shape[0])):
            gt_image = images[i]
            transform = transforms[i]
            H, W, _ = gt_image.shape
            R = transform[:3,:3]
            T = transform[:3,3]
            ray_dirs, ray_origins = get_rays(gt_image, f, R, T)  # H x W x 3
            ray_dirs = ray_dirs.reshape((-1,3))
            ray_origins = ray_origins.reshape((-1,3))

            ray_sample_ids = random.sample(range(ray_dirs.shape[0]), args.n_rays)  # args.n_rays samples

            ray_points, ts = generate_ray_points(
                                        ray_dirs[ray_sample_ids],
                                        ray_origins[ray_sample_ids], 
                                        args.n_ray_points, 
                                        n_frequencies=args.n_pose_frequencies
                                    )  # (n_rays*n_ray_points) x 3

            # execute the model
            rgbs = model(ray_points)  # model((n_rays*n_ray_points) x (3*2*n_freqs)) -> (n_rays*n_ray_points) x 4

            rgbs = rgbs.reshape((args.n_rays, args.n_ray_points, 4))  # n_rays x n_ray_points x 4

            train_image_values = volumetric_rendering(rgbs, ts, ray_dirs)
            gt_image_values = gt_image.reshape((-1,3))[ray_sample_ids]
            loss_this_iter = photometric_loss(gt_image_values, train_image_values)

            optimizer.zero_grad()
            loss_this_iter.backward()
            optimizer.step()
        
            # NOTE: from pytorch implementation of NeRF
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
#            decay_rate = 0.1
#            decay_steps = args.lrate_decay * 1000
#            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = new_lrate

    print("training is done!")



def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        print("need to give some argument!")
    return

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")
    parser.add_argument('--dataset_path',default='../data/lego/',help="dataset path")
    parser.add_argument('--mode',default=False,help="to train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--lrate_decay',default=25,help="decay learning rate")
    parser.add_argument('--n_ray_points',default=32,help="number of samples on a ray")
    parser.add_argument('--n_pose_frequencies',default=2,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_rays',default=4,help="number of rays to consider in an image")
    parser.add_argument('--max_iters',default=20000,help="number of max iterations for training")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)


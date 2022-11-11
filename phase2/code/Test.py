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
from torch.utils.tensorboard import SummaryWriter
import glob

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
    positions = [x]
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
    ray_directions = torch.stack([x, -y, -torch.ones_like(x)], axis=-1)  # H x W x 3
    # TODO
    # H*W x 3
    # 3 x H*W
    # R (3 x 3) * (3 x H*W)
    # 3 x H*W
    # (H*W) x 3
    # H x W x 3

    # apply rotation to the directions
    ray_directions = ray_directions[..., None, :]  # H x W x 1 x 3
    ray_directions = ray_directions * R  # H x W x 3 x 3 = H x W x 1 x 3 * 3 x 3
    ray_directions = torch.sum(ray_directions, -1) # H x W x 3
    ray_origins = torch.broadcast_to(T, ray_directions.shape) # H x W x 3
    return ray_directions, ray_origins

def generate_ray_points(ray_directions, ray_origins, n_ray_point_samples, t_near=2, t_far=6, N_rand_rays=None, n_frequencies=4, rand=False):
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

    if rand:
        # Inject uniform noise into sample space to make the sampling continuous.
        shape = list(ray_origins.shape[:-1]) + [n_ray_point_samples]
        noise = torch.tensor(np.random.uniform(size=shape) * (t_far - t_near) / n_ray_point_samples)
        ts = ts + noise

    rays = ray_origins[..., None, :] + ray_directions[..., None, :]*ts[..., None]  # (n_rays x n_samples x 3 
    points = rays.reshape((-1,3))  # (n_rays*n_samples) x 3
    points = encode_positions(points, n_frequencies)  # (n_rays*n_samples) x (3*4)
    return points, ts

def photometric_loss(gt_image, train_image):
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    img_loss = torch.mean((train_image - gt_image)**2)
    return img_loss

def find_and_load_latest_model(model, args):
    start_iter = 0

    files = glob.glob(args.checkpoint_path + '*.ckpt')
    
    latest_ckpt_file = max(files, key=os.path.getctime) if files else None

    if latest_ckpt_file and args.load_checkpoint:
        print(latest_ckpt_file)
        latest_ckpt = torch.load(latest_ckpt_file)

        start_iter = latest_ckpt_file.replace(args.checkpoint_path,'').replace('model_','').replace('.ckpt','')

        start_iter = int(start_iter)
        
        model.load_state_dict(latest_ckpt['model_state_dict'])
        print(f"Loaded latest checkpoint from {latest_ckpt_file} ....")
    else:
        print('New model initialized....')
    
    return start_iter

def render(gt_image, transform, f, model, is_full_render, args):
    H, W, _ = gt_image.shape
    R = transform[:3,:3]
    T = transform[:3,3]
    ray_dirs, ray_origins = get_rays(gt_image, f, R, T)  # H x W x 3
    ray_dirs = ray_dirs.reshape((-1,3))
    ray_origins = ray_origins.reshape((-1,3))

    ray_sample_ids = None
    if is_full_render:
        ray_sample_ids = range(ray_dirs.shape[0])
    else:
        ray_sample_ids = random.sample(range(ray_dirs.shape[0]), args.n_rays)  # args.n_rays samples

    ray_origins = ray_origins[ray_sample_ids]
    ray_dirs = ray_dirs[ray_sample_ids]
    gt_image_values = gt_image.reshape((-1,3))[ray_sample_ids]

    ray_points, ts = generate_ray_points(
                                ray_dirs,
                                ray_origins, 
                                args.n_ray_points, 
                                n_frequencies=args.n_pose_frequencies
                            )  # (n_rays*n_ray_points) x 3

    # execute the model
    rgbs = model(ray_points)  # model((n_rays*n_ray_points) x (3*2*n_freqs)) -> (n_rays*n_ray_points) x 4

    rgbs = rgbs.reshape(((len(ray_sample_ids)), args.n_ray_points, 4))  # n_rays x n_ray_points x 4

    train_image_values = volumetric_rendering(rgbs, ts, ray_dirs) # n_rays x 3
    return train_image_values, gt_image_values

def test(args):
    lego_dataset = NeRFDatasetLoader(args.dataset_path, args.mode)
    f, transforms, images = lego_dataset.get_full_data(device)
    
    # get input channels
    height, width, _ = images[0].shape
    input_channels = 3*2*args.n_pose_frequencies

    # initialize the model
    model = NeRF(input_channels, args.network_width).to(device)
    model.eval()
    image1 = images[0]
    transform1 = transforms[0]
    image_values, _ = render(image1, transform1, f, model, True, args) # n_rays x 3
    image = image_values.reshape(image1.shape)
    cv2.imwrite(f"image_test.png", image.detach().cpu().numpy())
    cv2.imwrite(f"image1.png", image1.detach().cpu().numpy())

    return


def train(args):
    # setup tensorboard
    writer = SummaryWriter(args.logs_path)

    lego_dataset = NeRFDatasetLoader(args.dataset_path, args.mode)
    f, transforms, images = lego_dataset.get_full_data(device)

    # get input channels
    height, width, _ = images[0].shape
    input_channels = 3*2*args.n_pose_frequencies

    # initialize the model
    model = NeRF(input_channels, args.network_width).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    # load the model if already exists
    start_iter = find_and_load_latest_model(model, args)

    model.train()
    
    print(f"starting iteration from: {start_iter}")
    for i_iter in tqdm(range(start_iter, args.max_iters)):
        
        loss_this_iter = 0

        for ith_view in tqdm(range(transforms.shape[0])):
            gt_image = images[ith_view]
            transform = transforms[ith_view]
            
            train_image_values, gt_image_values = render(gt_image, transform, f, model, False, args) #(n_rays, 3)
            loss_this_view = photometric_loss(gt_image_values, train_image_values)

            optimizer.zero_grad()
            loss_this_view.backward()
            optimizer.step()

            # accumulate the losses for this iteration
            loss_this_iter += loss_this_view

            
            # NOTE: from pytorch implementation of NeRF
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            #decay_rate = 0.1
            #decay_steps = args.lrate_decay * 1000
            #new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = new_lrate

            # Tensorboard
            writer.add_scalar('LossEveryView', loss_this_view, i_iter*transforms.shape[0] + ith_view)
            writer.flush()
        
        print(f"i_iter:{i_iter}, loss_this_iter:{loss_this_iter}")
        writer.add_scalar('LossEveryIter', loss_this_iter, i_iter)
        writer.flush()


        # Save checkpoint every some SaveCheckPoint's iterations
        if i_iter % args.save_ckpt_every_n_iters == 0:
            # Save the Model learnt in this epoch
            checkpoint_save_name =  args.checkpoint_path + os.sep + 'model_' + str(i_iter) + '.ckpt'
            torch.save({'iter': i_iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_this_iter}, checkpoint_save_name)

            # render 1st view and save for every checkpoint
            # model.eval()
            # image1 = images[0]
            # transform1 = transforms[0]
            # image_values, _ = render(image1, transform1, f, model, True, args) # n_rays x 3
            # image = image_values.reshape(image1.shape)
            # cv2.imwrite(f"../{args.images_folder}/image_{i_iter}.png", image.detach().cpu().numpy())
            # model.train()



        # validationBatch = GenerateBatch(TrainSet, valid_idx, TrainLabels, ImageSize, MiniBatchSize)
        # validation_result = model.validation_step(validationBatch)
        # Writer.add_scalar('ValidationLossEveryEpoch',validation_result['loss'],Epochs)
        # Writer.add_scalar('ValidationAccuracyEveryEpoch',validation_result['acc'],Epochs)
        # Writer.flush()
        

    print("training is done!")



def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        print("Broooooooo! need to give some argument man!")
    return

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")
    parser.add_argument('--dataset_path',default='../data/lego/',help="dataset path")
    parser.add_argument('--mode',default=False,help="to train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--lrate_decay',default=25,help="decay learning rate")
    parser.add_argument('--n_ray_points',default=64,help="number of samples on a ray")
    parser.add_argument('--n_pose_frequencies',default=2,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_rays',default=32*32*8,help="number of rays to consider in an image")
    parser.add_argument('--max_iters',default=200000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="../logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="../checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=False,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_every_n_iters',default=20,help="checkpoints path")
    parser.add_argument('--network_width', default=64,help="number of channels in the network")
    parser.add_argument('--images_folder', default="../images_folder/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)


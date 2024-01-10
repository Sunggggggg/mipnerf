import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from einops import rearrange
#
from config import mae_args_parser
# DDP
import torch.multiprocessing as mp
from set_multi_gpus import set_ddp
from torch.nn.parallel import DistributedDataParallel as DDP
# dataset
from dataset import load_nerf_synthetic_data, load_nerf_llff_data
#
from MAE import make_input, IMAGE_MAE, PATCH_MAE
# 
from MAE import image_plot, to8b

FIX = True 

def train(rank, world_size, args):
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # log 
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # save args   
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # save weight, fig
    os.makedirs(os.path.join(basedir, expname, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'figures'), exist_ok=True)

    # load dataset
    mae_input = args.mae_input
    assert np.sqrt(mae_input) % 1 == 0.0, "#of input should be square an integer"
    n = int(mae_input**0.5)

    if args.dataset_type == 'blender':
        train_imgs, train_poses, hwf, object_list = \
            load_nerf_synthetic_data(args.datadir, num_inputs=mae_input, scale=args.scale, white_bkgd=args.white_bkgd)
    elif args.dataset_type == 'LLFF':
        train_imgs, train_poses, hwf, object_list = \
            load_nerf_llff_data(args.datadir, num_inputs=mae_input, scale=args.scale, llffhold=8)

    print("Data load shape")
    print(f"image shape {train_imgs.shape}")
    print(f"poses shape {train_poses.shape}")

    H, W = int(hwf[0]), int(hwf[1])
    num_objects = len(object_list)
    print(f"#of Object {num_objects} \n {object_list}")

    # Plot images
    fig_path = os.path.join(basedir, expname, 'figures')
    for idx in range(num_objects):
        dir_path = os.path.join(fig_path, object_list[idx])
        os.makedirs(dir_path, exist_ok=True)

    train_imgs = make_input(train_imgs, args.emb_type, fig_path, object_list, n)     # [B, 3, N, H, W]

    # Model build
    if args.emb_type == "IMAGE" :
        mae = IMAGE_MAE
    else :
        mae = PATCH_MAE
    mae = mae(args, H, W).to(rank)
    optimizer = torch.optim.Adam(params=mae.parameters(), lr=args.lrate)
    
    # Move gpu
    train_imgs = torch.Tensor(train_imgs).to(rank)         # [B, 3, N, H, W]
    train_poses = torch.Tensor(train_poses).to(rank)       # [B, N, 4, 4]

    # if use multi gpus
    mae = DDP(mae, device_ids=[rank])
    print("Data parallel model with Multi gpus!")
    
    # Train
    start = 0
    epochs = args.epochs

    print("Train begin")
    start = start + 1
    for i in trange(start, epochs):
        # Train 
        object_shuffle_idx = torch.rand((num_objects)).argsort()
        imgs, poses = train_imgs[object_shuffle_idx], train_poses[object_shuffle_idx]
        
        loss, pred, mask = mae(imgs, poses)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # log/saving
        if i % args.i_figure == 0 :
            if args.emb_type == 'IMAGE':
                # pred : [B, N, (HxWx3)] 
                pred_img = rearrange(pred, 'B N (H W c) -> B c N H W', H=H, W=W, c=3)
                print(f"Reconstruct image {pred_img.shape}")

                for idx in range(num_objects) :
                    fig_name = f'pred_{i}.png'
                    png_path = os.path.join(basedir, expname, 'figures', object_list[object_shuffle_idx[idx]], fig_name)
                    image_plot(pred_img[idx], row=n, save_fig=png_path)
            else :
                # pred : [B, N, p1p2c]
                pred_img = rearrange(pred, 'B (n1 n2) (p1 p2 c) -> B (n1 p1) (n2 p2) c', n1=n, n2=n, p1=H, p2=W, c=3).detach().cpu().numpy()   # [o, Hxn, Wxn, 3]
                pred_img = to8b(pred_img)
                print(f"Reconstruct image {pred_img.shape}")

                for idx in range(num_objects) :
                    fig_name = f'pred_{i}.png'
                    png_path = os.path.join(basedir, expname, 'figures', object_list[object_shuffle_idx[idx]], fig_name)
                    
                    plt.imshow(pred_img[idx])
                    plt.axis('off')
                    plt.savefig(png_path)
                    plt.close()

        if i % args.i_weight == 0 : 
            model_name = f'mae_weight.tar'
            print("[SAVE] Model weights", model_name)
            torch.save({
            'model_state_dict' : mae.module.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(basedir, expname, 'weights', model_name))
            
        if i % args.i_print == 0 :
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.mean().item()}")

if __name__ == '__main__' :
    parser = mae_args_parser()
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
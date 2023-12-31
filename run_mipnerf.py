import os
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.optim as optim

import torch.multiprocessing as mp

# 
from config import config_parser
from set_multi_gpus import set_ddp
from dataset import load_data

# 
from scheduler import MipLRDecay
from loss import NeRFLoss
from model import MipNeRF

# 
from nerf_helper import *
from nerf_render import *


def train(rank, world_size, args):
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    # Load dataset
    images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test \
        = load_data(args.datadir, args.dataset_type, args.scale, args.testskip) # blender

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # logging dir
    loggin_dir = os.path.join(basedir, expname, 'eval.txt')

    model = MipNeRF(
        use_viewdirs=args.use_viewdirs,
        randomized=args.randomized,
        ray_shape=args.ray_shape,
        white_bkgd=args.white_bkgd,
        num_levels=args.num_levels,
        num_samples=args.N_samples,
        hidden=args.hidden,
        density_noise=args.density_noise,
        density_bias=args.density_bias,
        rgb_padding=args.rgb_padding,
        resample_padding=args.resample_padding,
        min_deg=args.min_deg,
        max_deg=args.max_deg,
        viewdirs_min_deg=args.viewdirs_min_deg,
        viewdirs_max_deg=args.viewdirs_max_deg,
        device=torch.device(rank),
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    scheduler = MipLRDecay(optimizer, lr_init=args.lr_init, lr_final=args.lr_final, 
                           max_steps=args.max_iters, lr_delay_steps=args.lr_delay_steps, 
                           lr_delay_mult=args.lr_delay_mult)

    # Loss func
    loss_func = NeRFLoss(args.coarse_weight_decay)

    # Move training data to GPU
    model.train()
    poses = torch.Tensor(poses).to(rank)
    render_poses = torch.Tensor(render_poses).to(rank)

    # Training hyperparams
    N_rand = args.N_rand
    max_iters = args.max_iters
    start = 0 + 1
    for i in trange(start, max_iters):
        # 1. Random select image
        img_i = np.random.choice(i_train)
        pose = poses[img_i, :3,:4]
        target = images[img_i]

        target = torch.Tensor(target).to(rank)
        pose = torch.Tensor(pose).to(rank)
        # 2. Generate rays
        rays_o, rays_d = get_rays(H, W, K, pose)
        radii = get_radii(rays_d)

        # 3. Random select rays
        if i < args.precrop_iters:
            dH = int(H//2 * args.precrop_frac)
            dW = int(W//2 * args.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                    torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                ), -1)
            if i == start:
                print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()        # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]   # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]   # (N_rand, 3)
        radii = radii[select_coords[:, 0], select_coords[:, 1]]     # (N_rand, 1)
        lossmult = torch.ones_like(radii)                              # (N_rand, 1)
        batch_rays = torch.stack([rays_o, rays_d], 0)                      # (2, N_rand, 3)
        target = target[select_coords[:, 0], select_coords[:, 1]]     # (N_rand, 3)
        
        # 4. Rendering 
        comp_rgbs, _, _ = render_mipnerf(H, W, K, chunk=args.chunk, netchunk=args.netchunk,
                                        mipnerf=model, rays=batch_rays, radii=radii, near=near, far=far,
                                        use_viewdirs=args.use_viewdirs, ndc=args.no_ndc)
        
        # 5. loss and update
        loss, (train_psnr_c, train_psnr_f) = loss_func(comp_rgbs, target, lossmult.to(rank))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i>0:
        # Rest is logging
        # if i%args.i_weights==0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'network_fn_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
        
        #if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                eval_psnr_c, eval_psnr_f = render_path(poses[i_test], hwf, K, args.chunk, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

            if rank == 0 :
                with open(loggin_dir, 'a') as file :
                    file.write(f"[ITER]{i:06d} PSNR_C : {eval_psnr_c:.3f}, PSNR_F : {eval_psnr_f:.3f}\n")

        #if i%args.i_print==0 and rank == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Total Loss: {train_psnr_c.item():.4f} PSNR: {train_psnr_f.item():.4f}")
        

if __name__ == '__main__' :
    parser = config_parser()
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))

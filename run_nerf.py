import os
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# 
from config import config_parser
from set_multi_gpus import set_ddp, myDDP
from dataset import load_data, sampling_pose
from metric import get_metric
# 
from loss import NeRFLoss
from model import NeRF
# 
from nerf_helper import *
from nerf_render import *

#
from MAE import IMAGE, PATCH, mae_input_format
from loss import MAELoss    

FIX = True  # Fix nerf training images

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
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # logging dir
    logdir = os.path.join(basedir, expname, 'eval.txt')

    # Build model
    model = NeRF(
        use_viewdirs=args.use_viewdirs,
        randomized=args.randomized,
        white_bkgd=args.white_bkgd,
        num_levels=args.num_levels,
        num_samples=args.N_samples,
        hidden=args.hidden,
        density_noise=args.density_noise,
        resample_padding=args.resample_padding,
        min_deg=args.min_deg, 
        max_deg=args.max_deg,
        viewdirs_min_deg=args.viewdirs_min_deg,
        viewdirs_max_deg=args.viewdirs_max_deg,
        device=torch.device(rank)
    )
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
    # Training hyperparams
    N_rand = args.N_rand
    max_iters = args.max_iters + 1
    start = 0 + 1
    nerf_weight_path = 'nerf_weights.tar'

    # Load pretrained model
    if args.nerf_weight != None :
        print("Load MipNeRF model weight :", args.nerf_weight)
        ckpt = torch.load(args.nerf_weight) # 

        model.load_state_dict(ckpt['network_fn_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        #scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        nerf_weight_path = 'nerf_tun_weights.tar'


    # Set multi gpus
    model = DDP(model, device_ids=[rank])
    
    # Loss func (Mip-NeRF)
    loss_func = NeRFLoss()
    
    #################################
    # MAE
    if args.mae_weight != None :
        # 1. Select few-shot
        nerf_input = args.nerf_input
        mae_input = args.mae_input
        if FIX :
            # Always same input index
            i_train = np.arange(nerf_input) + mae_input
        else :
            i_train = random.sample(list(i_train), nerf_input)
        
        print("train idx", i_train)
        print("Masking Ratio : %.4f"%(1-nerf_input/mae_input))
        # 2. Build MAE (Only Encoder+a part)
        if args.emb_type == "IMAGE" :
            encoder = IMAGE
        else :
            encoder = PATCH

        encoder = encoder(args, H, W).to(rank)

        print("Load MAE model weight :", args.mae_weight)
        ckpt = torch.load(args.mae_weight, map_location=f"cuda:{rank}")       # Use only one gpu
        encoder.load_state_dict(ckpt['model_state_dict'], strict=False)

        encoder = myDDP(encoder, device_ids=[rank])
        encoder.eval()

        train_images, train_poses = torch.tensor(images[i_train]), torch.tensor(poses[i_train])
        mae_input_images, mae_input_poses = mae_input_format(train_images, train_poses, nerf_input, mae_input, args.emb_type)
        mae_input_images = mae_input_images.type(torch.cuda.FloatTensor).to(rank)      # [1, 3, N, H, W]
        mae_input_poses = mae_input_poses.type(torch.cuda.FloatTensor).to(rank)        # [1, N, 4, 4]

        with torch.no_grad() :
            gt_feat = encoder(mae_input_images, mae_input_poses, mae_input, nerf_input)  #[1, N+1, D]
        print(f"Feature vector shape : {gt_feat.shape}")
        
        # 3. MAE loss
        mae_loss_func = MAELoss(args.mae_loss_func)

    #################################
    # Move training data to GPU
    model.train()
    poses = torch.Tensor(poses).to(rank)
    render_poses = torch.Tensor(render_poses).to(rank)


    for i in trange(start, max_iters):
        # 1. Random select image
        img_i = np.random.choice(i_train)
        pose = poses[img_i, :3,:4]
        target = images[img_i]

        target = torch.Tensor(target).to(rank)
        pose = torch.Tensor(pose).to(rank)
        # 2. Generate rays
        rays_o, rays_d = get_rays(H, W, K, pose)

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
        batch_rays = torch.stack([rays_o, rays_d], 0)                 # (2, N_rand, 3)
        target = target[select_coords[:, 0], select_coords[:, 1]]     # (N_rand, 3)
        
        # 4. Rendering 
        comp_rgbs, _, _ = render_nerf(H, W, K, chunk=args.chunk, netchunk=args.netchunk,
                                    nerf=model, rays=batch_rays, near=near, far=far,
                                    use_viewdirs=args.use_viewdirs, ndc=args.no_ndc)
        
        # 5. loss and update
        loss, (mse_loss_c, mse_loss_f), (train_psnr_c, train_psnr_f) = loss_func(comp_rgbs, target)

        # MAE
        if args.mae_weight :
            with torch.no_grad() :
                sampled_poses = sampling_pose(nerf_input, theta_range=[-180.+1.,180.-1.], phi_range=[-90., 0.], radius_range=[3.5, 4.5])
                rgbs = render_path(sampled_poses.to(rank), hwf, K, args.chunk, model, 
                                    near=near, far=far, use_viewdirs=args.use_viewdirs, no_ndc=args.no_ndc, progress_bar=False) # [N, 2, H, W, 3]
                rgbs = torch.tensor(rgbs)
                rgbs_c, rgbs_f = rgbs[:, 0], rgbs[:, 1]

                # Coarse
                rgbs_images, rgbs_poses = mae_input_format(rgbs_c, sampled_poses, nerf_input, mae_input, args.emb_type)
                rgbs_images = rgbs_images.type(torch.cuda.FloatTensor).to(rank)      # [1, 3, N, H, W] or # [1, 3, Hn, Wn]
                rgbs_poses = rgbs_poses.type(torch.cuda.FloatTensor).to(rank)        # [1, N, 4, 4]
                rendered_feat = encoder(rgbs_images, rgbs_poses, mae_input, nerf_input)
                object_loss_c = mae_loss_func(gt_feat[:, 1:, :], rendered_feat[:, 1:, :])
                object_loss_c = object_loss_c * args.loss_lam_c

                # Fine
                rgbs_images, rgbs_poses = mae_input_format(rgbs_f, sampled_poses, nerf_input, mae_input, args.emb_type)
                rgbs_images = rgbs_images.type(torch.cuda.FloatTensor).to(rank)      # [1, 3, N, H, W]
                rgbs_poses = rgbs_poses.type(torch.cuda.FloatTensor).to(rank)        # [1, N, 4, 4]
                rendered_feat = encoder(rgbs_images, rgbs_poses, mae_input, nerf_input)
                object_loss_f = mae_loss_func(gt_feat[:, 1:, :], rendered_feat[:, 1:, :])
                object_loss_f = object_loss_f * args.loss_lam_f

                loss += (object_loss_f + object_loss_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Rest is logging
        if i%args.i_weights==0 and i > 0:
            path = os.path.join(basedir, expname, nerf_weight_path)
            torch.save({
                'network_fn_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict' : scheduler.state_dict()
            }, path)
            print('Saved checkpoints at', path)
        

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                rgbs = render_path_nerf(poses[i_test], hwf, K, args.chunk, model, 
                                    near=near, far=far, use_viewdirs=args.use_viewdirs, no_ndc=args.no_ndc, 
                                    gt_imgs=images[i_test], savedir=testsavedir)
                eval_psnr, eval_ssim, eval_lpips = get_metric(rgbs[:, -1], images[i_test], None, torch.device(rank))    # Use fine model
            if rank == 0 :
                with open(logdir, 'a') as file :
                    file.write(f"{i:06d}-iter PSNR : {eval_psnr:.3f}, SSIM : {eval_ssim:.3f}, LPIPS : {eval_lpips:.3f}\n")
            print('Saved test set')

        if i%args.i_print==0 and rank == 0 :
            tqdm.write(f"[MSE]      C_Loss: {mse_loss_c.item():.6f}\t f_Loss: {mse_loss_f.item():.6f}")
            tqdm.write(f"[COSINE]   C_Loss: {object_loss_c.item():.6f}\t f_Loss: {object_loss_f.item():.6f}")
            #tqdm.write(f"[TRAIN]    Iter: {i} Total Loss: {loss.item():.6f} PSNR: {train_psnr_f.item():.4f} LR: {float(scheduler.get_last_lr()[-1]):.6f}")
            tqdm.write(f"[TRAIN]    Iter: {i:06d} Total Loss: {loss.item():.6f} PSNR: {train_psnr_f.item():.4f}")
        
        # logging
        with open(os.path.join(basedir, expname, 'log.txt'), 'a') as f :
            f.write(f"[MSE]      C_Loss: {mse_loss_c.item():.6f}\t f_Loss: {mse_loss_f.item():.6f}\n")
            f.write(f"[COSINE]   C_Loss: {object_loss_c.item():.6f}\t f_Loss: {object_loss_f.item():.6f}\n")
            f.write(f"[TRAIN]    Iter: {i:06d} Total Loss: {loss.item():.6f} PSNR: {train_psnr_f.item():.4f}\n")

if __name__ == '__main__' :
    parser = config_parser()
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))

import os
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from dataset import blender_sampling_pose, llff_sampling_pose
from .visualize import image_plot

def make_input(imgs, emb_type, fig_path, object_list, n=5, save_fig=True):
    """
    imgs        [B, N, H, W, 3]
    """
    # Make input format
    if emb_type == 'IMAGE':
        """ O -> B
        imgs.shape should be [B, N, H, W, 3] -> [B, 3, N, H, W]
        """
        print("Use Image embedding")

        imgs = imgs.transpose(0, -1, 1, 2, 3)

        if save_fig :
            for idx, _object in enumerate(object_list) :
                png_path = os.path.join(fig_path, _object, 'input.png')
                image_plot(imgs[idx], row=n, save_fig=png_path)

    else : # args.emb_type == 'PATCH'
        """
        imgs.shape should be [B, N, H, W, 3] -> [B, 3, H*n, W*n]
        """
        print("Use Patch embedding")
        imgs = rearrange(imgs, 'O (n1 n2) H W c -> O (n1 H) (n2 W) c', c=3, n1=n, n2=n)
        
        # Plot images
        if save_fig :
            for idx, _object in enumerate(object_list) :
                png_path = os.path.join(fig_path, _object, 'input.png')
                plt.imshow(imgs[idx])
                plt.axis('off')
                plt.savefig(png_path)
                plt.close()
        imgs = imgs.transpose(0, 3, 1, 2)       # [B, 3, H*n, W*n]
            
    return imgs

def mae_input_format(imgs, poses, nerf_input, mae_input, emb_type='IMAGE', sampling_pose_function=None):
    """ NeRF input format with MAE input format (F = nerf_input / N = mae_input)
    args
    imgs  (torch) [F, H, W, 3]
    poses (torch) [F, 4, 4] 

    return
        Cam_pos_encoding = True
        imgs        [B, 3, N, H, W]
        poses       [B, N, 4, 4]

        Cam_pos_encoding = False
        imgs        [B, 3, Hxn, Wxn]
        poses       [B, N, 4, 4] 
    """
    
    rand_pose = sampling_pose_function(mae_input-nerf_input)

    # 
    imgs = torch.cat([imgs, torch.zeros((mae_input-nerf_input, *imgs.shape[1:]))], dim=0)       
    poses = torch.cat([poses, rand_pose], dim=0)
    
    #
    if emb_type == 'IMAGE' :
        imgs = imgs.permute(3, 0, 1, 2).unsqueeze(0)    # [1, 3, N, H, W]
        poses = poses.unsqueeze(0)                      # [1, N, 4, 4]
    else :
        n = int(mae_input**0.5)
        imgs = imgs.unsqueeze(0)
        imgs = rearrange(imgs, 'B (n1 n2) H W c -> B c (n1 H) (n2 W)', c=3, n1=n, n2=n)
        poses = poses.unsqueeze(0)       # [1, N, 4, 4] 

    return imgs, poses
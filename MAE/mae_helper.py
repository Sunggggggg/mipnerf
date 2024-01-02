import torch
from einops import rearrange

def mae_input_format(imgs, poses, nerf_input, mae_input, emb_type='IMAGE'):
    """ NeRF input format with MAE input format (F = nerf_input / N = mae_input)
    args
    imgs  (numpy) [F, H, W, 3]
    poses (numpy) [F, 4, 4] 

    return
        Cam_pos_encoding = True
        imgs        [B, 3, N, H, W]
        poses       [B, N, 4, 4]

        Cam_pos_encoding = False
        imgs        [B, 3, Hxn, Wxn]
        poses       [B, N, 4, 4] 
    """
    imgs = torch.tensor(imgs)    # [F, H, W, 3]  
    poses = torch.tensor(poses)  # [F, 4, 4]

    # Only use [:nerf_input]  
    imgs = torch.cat([imgs, torch.zeros((mae_input-nerf_input, *imgs.shape[1:]))], dim=0)       
    poses = torch.cat([poses, torch.zeros((mae_input-nerf_input, *poses.shape[1:]))], dim=0)
    
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
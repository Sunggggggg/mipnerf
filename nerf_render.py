import os
from tqdm import tqdm
import imageio
import torch
from nerf_helper import *

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def render_mipnerf(H, W, K, chunk=1024*16, netchunk=1024*32,
                   mipnerf=None, rays=None, radii=None, c2w=None, near=0., far=1.,
                   use_viewdirs=True, ndc=False):
    """
    Return
    all_comp_rgbs   # [2, N_rand, 3]
    all_distances   # [2, N_rand, 1]
    all_accs        # [2, N_rand, 1]
    """
    if c2w is not None:
        # Only use rendering
        rays_o, rays_d = get_rays(H, W, K, c2w)
        radii = get_radii(rays_d)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1,3]).float()      # [N_rand, 3]
    rays_d = torch.reshape(rays_d, [-1,3]).float()      # [N_rand, 3]
    radii = torch.reshape(radii, [-1,1]).float()        # [N_rand, 1]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) # (H*W, 1)
    rays = torch.cat([rays_o, rays_d, near, far, radii], -1)   
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)          # (H*W, 3 + 3 + 1 + 1 + 1 + 3)
    
    all_comp_rgbs, all_distances, all_accs = [], [], []
    for i in range(0, rays.shape[0], chunk):
        comp_rgbs, distances, accs = mipnerf(rays[i:i+chunk])
        all_comp_rgbs.append(comp_rgbs)     # [2, chunk, 3]
        all_distances.append(distances)     # [2, chunk, 1]
        all_accs.append(accs)               # [2, chunk, 1]

    all_comp_rgbs = torch.cat(all_comp_rgbs, 1) # [2, N_rand, 3]
    all_distances = torch.cat(all_distances, 1) # [2, N_rand, 1]
    all_accs = torch.cat(all_accs, 1)           # [2, N_rand, 1]

    return all_comp_rgbs, all_distances, all_accs

def render_nerf(H, W, K, chunk=1024*16, netchunk=1024*32,
                nerf=None, rays=None, c2w=None, near=0., far=1.,
                use_viewdirs=True, ndc=False):
    """
    Return
    all_comp_rgbs   # [2, N_rand, 3]
    all_distances   # [2, N_rand, 1]
    all_accs        # [2, N_rand, 1]
    """
    if c2w is not None:
        # Only use rendering
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1,3]).float()      # [N_rand, 3]
    rays_d = torch.reshape(rays_d, [-1,3]).float()      # [N_rand, 3]
    
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) # (H*W, 1)
    rays = torch.cat([rays_o, rays_d, near, far], -1)   
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)          # (H*W, 3 + 3 + 1 + 1 + 3)
    
    all_comp_rgbs, all_distances, all_accs = [], [], []
    for i in range(0, rays.shape[0], chunk):
        comp_rgbs, distances, accs = nerf(rays[i:i+chunk])
        all_comp_rgbs.append(comp_rgbs)     # [2, chunk, 3]
        all_distances.append(distances)     # [2, chunk, 1]
        all_accs.append(accs)               # [2, chunk, 1]

    all_comp_rgbs = torch.cat(all_comp_rgbs, 1) # [2, N_rand, 3]
    all_distances = torch.cat(all_distances, 1) # [2, N_rand, 1]
    all_accs = torch.cat(all_accs, 1)           # [2, N_rand, 1]

    return all_comp_rgbs, all_distances, all_accs

@torch.no_grad()
def render_sample_path(render_poses, hwf, K, chunk, mipnerf, 
                near=0., far=1., use_viewdirs=True, no_ndc=False, 
                gt_imgs=None, savedir=None, render_factor=0, progress_bar=True):
    """ Rendering only
    Args
    render_poses (tensor) : [N, 4, 4]
    
    Return
    rgbs (numpy) float32 : [N, 2, H, W, 3]
    """
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    if progress_bar :
        for i, c2w in enumerate(tqdm(render_poses)):
            rgb, _, _= render_mipnerf(H, W, K, chunk=chunk, 
                                    mipnerf=mipnerf, c2w=c2w[:3,:4], near=near, far=far,
                                    use_viewdirs=use_viewdirs, ndc=no_ndc)
            rgb = torch.reshape(rgb, [2, H, W, 3])

            if savedir is not None:
                rgb8 = to8b(rgb[-1].cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
        
            rgbs.append(rgb.cpu().numpy())
    else : 
        for i, c2w in enumerate(render_poses):
            rgb, _, _= render_mipnerf(H, W, K, chunk=chunk, 
                                            mipnerf=mipnerf, c2w=c2w[:3,:4], near=near, far=far,
                                            use_viewdirs=use_viewdirs, ndc=no_ndc)
            rgb = torch.reshape(rgb, [2, H, W, 3])

            if savedir is not None:
                rgb8 = to8b(rgb[-1].cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
        
            rgbs.append(rgb.cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    return rgbs

def render_path(render_poses, hwf, K, chunk, mipnerf, 
                near=0., far=1., use_viewdirs=True, no_ndc=False, 
                gt_imgs=None, savedir=None, render_factor=0, progress_bar=True):
    """ Rendering only
    Args
    render_poses (tensor) : [N, 4, 4]
    
    Return
    rgbs (numpy) float32 : [N, 2, H, W, 3]
    """
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    if progress_bar :
        for i, c2w in enumerate(tqdm(render_poses)):
            rgb, _, _= render_mipnerf(H, W, K, chunk=chunk, 
                                    mipnerf=mipnerf, c2w=c2w[:3,:4], near=near, far=far,
                                    use_viewdirs=use_viewdirs, ndc=no_ndc)
            rgb = torch.reshape(rgb, [2, H, W, 3])

            if savedir is not None:
                rgb8 = to8b(rgb[-1].cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
        
            rgbs.append(rgb.cpu().numpy())
    else : 
        for i, c2w in enumerate(render_poses):
            rgb, _, _= render_mipnerf(H, W, K, chunk=chunk, 
                                            mipnerf=mipnerf, c2w=c2w[:3,:4], near=near, far=far,
                                            use_viewdirs=use_viewdirs, ndc=no_ndc)
            rgb = torch.reshape(rgb, [2, H, W, 3])

            if savedir is not None:
                rgb8 = to8b(rgb[-1].cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
        
            rgbs.append(rgb.cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    return rgbs

def render_path_nerf(render_poses, hwf, K, chunk, nerf, 
                near=0., far=1., use_viewdirs=True, no_ndc=False, 
                gt_imgs=None, savedir=None, render_factor=0, progress_bar=True):
    """ Rendering only
    Args
    render_poses (tensor) : [N, 4, 4]
    
    Return
    rgbs (numpy) float32 : [N, 2, H, W, 3]
    """
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    if progress_bar :
        for i, c2w in enumerate(tqdm(render_poses)):
            rgb, _, _= render_nerf(H, W, K, chunk=chunk, 
                                    nerf=nerf, c2w=c2w[:3,:4], near=near, far=far,
                                    use_viewdirs=use_viewdirs, ndc=no_ndc)
            rgb = torch.reshape(rgb, [2, H, W, 3])

            if savedir is not None:
                rgb8 = to8b(rgb[-1].cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
        
            rgbs.append(rgb.cpu().numpy())
    else : 
        for i, c2w in enumerate(render_poses):
            rgb, _, _= render_nerf(H, W, K, chunk=chunk, 
                                    nerf=nerf, c2w=c2w[:3,:4], near=near, far=far,
                                    use_viewdirs=use_viewdirs, ndc=no_ndc)
            rgb = torch.reshape(rgb, [2, H, W, 3])

            if savedir is not None:
                rgb8 = to8b(rgb[-1].cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
        
            rgbs.append(rgb.cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    return rgbs

def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function For Mip-NeRF

    Args:
    rgb         : [N_rays, N_samples, 3]
    density     : [N_rays, N_samples, 1]    # Already activate
    t_vals      : [N_rays, N_samples + 1]
    dirs        : [N_rays, 3]
    white_bkgd  : 

    Returns:
    comp_rgb    : [N_rays, N_samples, 3]
    distance    : [N_rays]
    acc         : [N_rays]
    weights     : [N_rays, N_samples]
    alpha       : [N_rays, N_samples]    
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])      # [N_rays, N_samples] 
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]             # [N_rays, N_samples] 
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1) # [N_rays, N_samples]
    density_delta = density[..., 0] * delta                  # [N_rays, N_samples, 1]

    alpha = 1 - torch.exp(-density_delta)                    #
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights, alpha

def volumetric_rendering_nerf(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
    rgb         : [N_rays, N_samples, 3]
    density     : [N_rays, N_samples, 1]    # Already activate
    t_vals      : [N_rays, N_samples]
    dirs        : [N_rays, 3]
    white_bkgd  : 

    Return:
    comp_rgb    : [N_rays, N_samples, 3]
    distance    : [N_rays]
    acc         : [N_rays]
    weights     : [N_rays, N_samples]
    alpha       : [N_rays, N_samples]
    """
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]    # [N_rays, N_samples-1] 

    # Append 100000
    t_dists = torch.cat([t_dists, torch.Tensor([1e10]).expand(t_dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)                         # [N_rays, N_samples]
    density_delta = density[..., 0] * delta                                                 # [N_rays, N_samples]
    
    alpha = 1 - torch.exp(-density_delta)               # [N_rays, N_samples]
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),       # [N_rays, 1]
        torch.cumsum(density_delta[..., :-1], dim=-1)   # [N_rays, N_samples-1]
    ], dim=-1)) 
    weights = alpha * trans                             # [N_rays, N_samples] 

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)   # [N_rays, N_samples, 3] 
    acc = weights.sum(dim=-1)                           # [N_rays]
    distance = (weights * t_vals).sum(dim=-1) / acc     # [N_rays]
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights, alpha
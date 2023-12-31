import torch
from nerf_helper import *

def render_mipnerf(H, W, K, chunk=1024*32, 
                   mipnerf=None, rays=None, radii=None, c2w=None, near=0., far=1.,
                   use_viewdirs=True, ndc=False):
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
        all_comp_rgbs.append(comp_rgbs)     # [2, N_rand, 3]
        all_distances.append(distances)     # [2, N_rand, 1]
        all_accs.append(accs)               # [2, N_rand, 1]

    all_comp_rgbs = torch.cat(all_comp_rgbs, 1)
    all_distances = torch.cat(all_distances, 1)
    all_accs = torch.cat(all_accs, 1)

    return all_comp_rgbs, all_distances, all_accs

def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
    rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, num_samples, 1].
    t_vals: torch.tensor(float32), [batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, 3].
    white_bkgd: bool.

    Returns:
    comp_rgb: torch.tensor(float32), [batch_size, 3].
    disp: torch.tensor(float32), [batch_size].
    acc: torch.tensor(float32), [batch_size].
    weights: torch.tensor(float32), [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
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
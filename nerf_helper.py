import torch
import numpy as np

def get_radii(rays_d):
    """
    args
        rays_d :    [H, W, 3]

    return
        radii  :    [H, W, 1]
    """
    dx = torch.sqrt(torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    radii = dx[..., None] * 2 / 12**0.5

    return radii
# Ray helpers
def get_rays(H, W, K, c2w):
    """ All rays from origin, rays_o, rays_d is dir vectors
    """
    device = c2w.device

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(device)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    """ All rays from origin, rays_o, rays_d is dir
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates.
    d           : [N_rays, 3]
    t_mean      : [N_rays, N_samples]
    t_var       : [N_rays, N_samples]
    r_var       : [N_rays, N_samples]
    """
    mean = d[..., None, :] * t_mean[..., None]      # [N_rays, N_samples, 3] 

    d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True) + 1e-10  # [N_rays, 1]

    if diag:
        d_outer_diag = d ** 2                                               # [N_rays, 3]
        null_outer_diag = 1 - d_outer_diag / d_mag_sq                       # [N_rays, 3]
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]          # [N_rays, N_samples, 3]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]      # [N_rays, N_samples, 3]
        cov_diag = t_cov_diag + xy_cov_diag                                 # [N_rays, N_samples, 3]
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]         # [B, 3, 3]
        
        eye = torch.eye(d.shape[-1], device=d.device)       # [3, 3]
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]   # [B, 3, 3]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]           # [B, N, 3, 3]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]       # [B, N, 3, 3]
        cov = t_cov + xy_cov
    return mean, cov

def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """ Sampling conical frustum region
    
    d : directions from camera # [N_rays, 3]
    t0 :                       # [N_rays, N_samples]
    t1 :                       # [N_rays, N_samples]
    base_radius : Cone radius  # [N_rays, 1]
    
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)
    
def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """ Sampling cylinder region
    
    d : directions from camera # [N_rays, 3]
    t0 :                       # [N_rays, N_samples]
    t1 :                       # [N_rays, N_samples]
    base_radius : Cone radius  # [N_rays, 1]
    
    """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)

def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    """ Sampling along the cone or cylinder

    z_vals  : real pts along the rays   # [N_rays, N_samples+1]

    origins : Camera origins            # [N_rays, 3]
    directions : directions from camera # [N_rays, 3]
    radii : Cone radius                 # [N_rays, 1]
    ray_shape : 'cone' OR 'cylinder'   

    return
    means, covs : mean and covs         # [N_rays, N_samples, 1]
    """
    # 
    t0 = t_vals[..., :-1]           # [N_rays, N_samples]
    t1 = t_vals[..., 1:]            # [N_rays, N_samples]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """
    bins            : [B, N+1]
    weights         : [B, N]
    num_samples     : [N]
    randomized      :
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)       # [B, 1]
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum          # [B, N]
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)   # [B, N-1]
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)
 
    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

def sorted_piecewise_constant_pdf_nerf(bins, weights, num_samples, randomized):
    """
    bins            : [B, N]
    weights         : [B, N]
    num_samples     : [N]
    randomized      :
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)       # [B, 1]
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum          # [B, N]
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)   # [B, N-1]
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device), cdf], dim=-1) # [B, N+1] 
 
    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, lindisp, ray_shape):
    """Stratified sampling along the rays.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      num_samples: int.
      near: torch.tensor, [batch_size, 1], near clip.
      far: torch.tensor, [batch_size, 1], far clip.
      randomized: bool, use randomized stratified sampling.
      lindisp: bool, sampling linearly in disparity rather than depth.

    Returns:
      t_vals: torch.tensor, [batch_size, num_samples+1], sampled z values.
      means: torch.tensor, [batch_size, num_samples, 3], sampled means.
      covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_samples + 1,  device=origins.device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    return t_vals, (means, covs)
    
def resample_along_rays(origins, directions, radii, t_vals, weights, randomized, stop_grad, resample_padding, ray_shape):
    """Resampling.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      weights: torch.tensor(float32), weights for t_vals
      randomized: bool, use randomized samples.
      stop_grad: bool, whether or not to backprop through sampling.
      resample_padding: float, added to the weights before normalizing.

    Returns:
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      points: torch.tensor(float32), [batch_size, num_samples, 3].
    """
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_vals,
                weights,
                t_vals.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_vals,
            weights,
            t_vals.shape[-1],
            randomized,
        )
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)

# def resample_along_rays(origins, directions, radii, t_vals, weights, N_importance, randomized, ray_shape):
#     """Resampling.
#     origins         : [N_rays, 3]
#     directions      : [N_rays, 3]
#     t_vals          : [N_rays, N_samples+1]
#     N_importance    : 
#     weights         : [N_rays, N_samples]
#     """
#     t_vals_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])     # [N_rays, N_samples]
    
#     weights = weights + 1e-5 # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)        # [N_rays, N_samples]
#     cdf = torch.cumsum(pdf, -1)                                 # [N_rays, N_samples]
#     cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)   # [N_rays, N_samples+1]

#     # Take uniform samples
#     if randomized :
#         u = torch.linspace(0., 1., steps=N_importance)             # 
#         u = u.expand(list(cdf.shape[:-1]) + [N_importance])        # [N_rays, N_importance] 
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_importance])      # [N_rays, N_importance] 

#     u = u.contiguous()
#     inds = torch.searchsorted(cdf, u, right=True)                   # [N_rays, N_importance] 
#     below = torch.max(torch.zeros_like(inds-1), inds-1)
#     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # [N_rays, N_importance, 2]
    
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]                   # [N_rays, N_importance, N_samples+1]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)             # [N_rays, N_importance, 2]
#     bins_g = torch.gather(t_vals.unsqueeze(1).expand(matched_shape), 2, inds_g)         # [N_rays, N_importance, 2]

#     denom = (cdf_g[...,1]-cdf_g[...,0])
#     denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
#     t = (u-cdf_g[...,0])/denom
#     new_t_vals = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])                      # [N_rays, N_importance]

#     new_t_vals, _ = torch.sort(torch.cat([t_vals, new_t_vals], -1), -1)                 # [N_rays, N_importance+N_samples]
#     means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)          # 
#     return new_t_vals, (means, covs)

def sample_along_rays_nerf(origins, directions, num_samples, near, far, randomized, lindisp):
    """
    origins         : [N_rays, 3]
    directions      : [N_rays, 3]
    num_samples     : N_samples
    near, far       : [N_rays, 1]

    Return
    t_vals          : [N_rays, N_samples]
    pts             : [N_rays, N_samples, 3]
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_samples,  device=origins.device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples])
    pts = origins[..., None, :] + directions[...,None,:] * t_vals[...,:,None] 
    return t_vals, pts
    
def resample_along_rays_nerf(origins, directions, t_vals, weights, N_importance, randomized):
    """Resampling.
    origins         : [N_rays, 3]
    directions      : [N_rays, 3]
    t_vals          : [N_rays, N_samples]
    N_importance    : 
    weights         : [N_rays, N_samples]
    """
    t_vals_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])     # [N_rays, N_samples-1]
    weights = weights[..., 1:-1]                                # [N_rays, N_samples-2], depadding

    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)        # [N_rays, N_samples-2]
    cdf = torch.cumsum(pdf, -1)                                 # [N_rays, N_samples-2]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)   # [N_rays, N_samples-1]

    # Take uniform samples
    if randomized :
        u = torch.linspace(0., 1., steps=N_importance)             # 
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])        # [N_rays, N_importance] 
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance])      # [N_rays, N_importance] 

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)                   # [N_rays, N_importance] 
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # [N_rays, N_importance, 2]

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]                   # [N_rays, N_importance, N_samples-1]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)             # [N_rays, N_importance, 2]
    bins_g = torch.gather(t_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)     # [N_rays, N_importance, 2]

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    new_t_vals = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])                 # [N_rays, N_importance]

    new_t_vals, _ = torch.sort(torch.cat([t_vals, new_t_vals], -1), -1)                 # [N_rays, N_importance+N_samples]
    pts = origins[..., None, :] + directions[...,None,:] * new_t_vals[...,:,None]       # [N_rays, N_importance+N_samples, 3]
    return new_t_vals, pts
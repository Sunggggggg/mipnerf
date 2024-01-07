import torch
import torch.nn as nn

from nerf_helper import sample_along_rays, resample_along_rays, sample_along_rays_nerf, resample_along_rays_nerf
from nerf_render import volumetric_rendering, volumetric_rendering_nerf

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        '''
        Args
        x : [B, N, 3]
        y : [B, N, 3]

        Return
        x_ret : [B, N, E]      (max_deg-min_deg)*(sin, cos)*(x,y,z) = E
        y_ret : [B, N, E]
        '''
        shape = list(x.shape[:-1]) + [-1]   # [B, N, -1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape) # [B, N, L, 3] -> [B, N, 3L] 
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)          # [B, N, 6L]
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret

# class MipNeRF(nn.Module):
#     def __init__(self,
#                  use_viewdirs=True,
#                  randomized=False,
#                  ray_shape="cone",
#                  white_bkgd=True,
#                  num_levels=2,
#                  N_samples=64,
#                  N_importance=128,
#                  hidden=256,
#                  density_noise=1, density_bias=-1,
#                  rgb_padding=0.001,
#                  resample_padding=0.01,
#                  min_deg=0, max_deg=16,
#                  viewdirs_min_deg=0, viewdirs_max_deg=4,
#                  device=torch.device("cpu"),
#                  return_raw=False
#                  ):
#         super(MipNeRF, self).__init__()
#         self.use_viewdirs = use_viewdirs
#         self.init_randomized = randomized
#         self.randomized = randomized
#         self.ray_shape = ray_shape
#         self.white_bkgd = white_bkgd
#         self.num_levels = num_levels
#         self.N_samples = N_samples
#         self.N_importance = N_importance
#         self.density_input = (max_deg - min_deg) * 3 * 2
#         self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
#         self.density_noise = density_noise
#         self.rgb_padding = rgb_padding
#         self.resample_padding = resample_padding
#         self.density_bias = density_bias
#         self.hidden = hidden
#         self.device = device
#         self.return_raw = return_raw
#         self.density_activation = nn.Softplus()

#         self.positional_encoding = PositionalEncoding(min_deg, max_deg)
#         self.density_net0 = nn.Sequential(
#             nn.Linear(self.density_input, hidden),
#             nn.ReLU(True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(True),
#         )
#         self.density_net1 = nn.Sequential(
#             nn.Linear(self.density_input + hidden, hidden),
#             nn.ReLU(True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(True),
#         )
#         self.final_density = nn.Sequential(
#             nn.Linear(hidden, 1),
#         )

#         input_shape = hidden
#         if self.use_viewdirs:
#             input_shape = hidden // 2

#             self.rgb_net0 = nn.Sequential(
#                 nn.Linear(hidden, hidden)
#             )
#             self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
#             self.rgb_net1 = nn.Sequential(
#                 nn.Linear(hidden + self.rgb_input, input_shape),
#                 nn.ReLU(True),
#             )
#         self.final_rgb = nn.Sequential(
#             nn.Linear(input_shape, 3),
#             nn.Sigmoid()
#         )
#         _xavier_init(self)
#         self.to(device)

#     def forward(self, ray_batch):
#         comp_rgbs = []
#         distances = []
#         accs = []

#         rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]                 # [N_rays, 3] each
#         bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])                # [N_rays, 1, 2]
#         near, far = bounds[...,0], bounds[...,1]                            # [N_rays, 1]
#         radii = torch.reshape(ray_batch[..., 8], [-1, 1])                   # [N_rays, 1]
#         view_dirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None   # [N_rays, 3]
        
#         for l in range(self.num_levels):
#             # sample
#             if l == 0:
#                 num_samples = self.N_samples
#                 t_vals, (mean, var) = sample_along_rays(rays_o, rays_d, radii, num_samples,
#                                                         near, far, randomized=self.randomized, lindisp=False,
#                                                         ray_shape=self.ray_shape)
#             else:
#                 num_samples = self.N_samples + self.N_importance
#                 t_vals, (mean, var) = resample_along_rays(rays_o, rays_d, radii, t_vals.to(rays_o.device),
#                                                           weights.to(rays_o.device), randomized=self.randomized,
#                                                           stop_grad=True, resample_padding=self.resample_padding,
#                                                           ray_shape=self.ray_shape)
#             # t_vals : [N_rays, N_samples+1]
#             # mean, var : [N_rays, N_samples, 3]
            
#             # do integrated positional encoding of samples
#             samples_enc = self.positional_encoding(mean, var)[0]
#             samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])  # [N_rays*N_samples, 96]

#             # predict density
#             new_encodings = self.density_net0(samples_enc)                  # [N_rays*N_samples, 256]
#             new_encodings = torch.cat((new_encodings, samples_enc), -1)     # [N_rays*N_samples, 256+96]
#             new_encodings = self.density_net1(new_encodings)                # [N_rays*N_samples, 256]
#             raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1)) # [N_rays, N_samples, 1]
            
#             # predict rgb
#             if self.use_viewdirs:
#                 #  do positional encoding of viewdirs
#                 viewdirs = self.viewdirs_encoding(view_dirs.to(self.device))             # [N_rays, 27]
#                 viewdirs = torch.cat((viewdirs, view_dirs.to(self.device)), -1)          # [N_rays, 30]
#                 viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))    # [N_rays, N_samples, 30]
#                 viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))                    # [N_rays*N_samples, 30]
#                 new_encodings = self.rgb_net0(new_encodings)                             # [N_rays*N_samples, 256]
#                 new_encodings = torch.cat((new_encodings, viewdirs), -1)                 # [N_rays*N_samples, 30+256]
#                 new_encodings = self.rgb_net1(new_encodings)                             # [N_rays*N_samples, 286]
#             raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))   # [N_rays, N_samples, 3]
            
#             # Add noise to regularize the density predictions if needed.
#             if self.randomized and self.density_noise:
#                 raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

#             # volumetric rendering
#             rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
#             density = self.density_activation(raw_density + self.density_bias)
#             comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals, rays_d.to(rgb.device), self.white_bkgd)
#             comp_rgbs.append(comp_rgb)
#             distances.append(distance)
#             accs.append(acc)
#         if self.return_raw:
#             raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
#             # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
#             return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
#         else:
#             # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
#             return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

class MipNeRF(nn.Module):
    def __init__(self,
                 use_viewdirs=True,
                 randomized=False,
                 ray_shape="cone",
                 white_bkgd=True,
                 num_levels=2,
                 N_samples=64,
                 N_importance=128,
                 hidden=256,
                 density_noise=1, density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0, max_deg=16,
                 viewdirs_min_deg=0, viewdirs_max_deg=4,
                 device=torch.device("cpu"),
                 return_raw=False
                 ):
        super(MipNeRF, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.density_input = (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.hidden = hidden
        self.device = device
        self.return_raw = return_raw
        self.density_activation = nn.Softplus()

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        input_shape = hidden
        if self.use_viewdirs:
            input_shape = hidden // 2

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, input_shape),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)
        self.to(device)

    def forward(self, ray_batch):
        comp_rgbs = []
        distances = []
        accs = []

        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]                 # [N_rays, 3] each
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])                # [N_rays, 1, 2]
        near, far = bounds[...,0], bounds[...,1]                            # [N_rays, 1]
        radii = torch.reshape(ray_batch[..., 8], [-1, 1])                   # [N_rays, 1]
        view_dirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None   # [N_rays, 3]
        
        for l in range(self.num_levels):
            # sample
            if l == 0:
                num_samples = self.N_samples
                t_vals, (mean, var) = sample_along_rays(rays_o, rays_d, radii, num_samples,
                                                        near, far, randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:
                num_samples = self.N_samples + self.N_importance
                t_vals, (mean, var) = resample_along_rays(rays_o, rays_d, radii, t_vals.to(rays_o.device),
                                                          weights.to(rays_o.device), self.N_importance, 
                                                          randomized=self.randomized, ray_shape=self.ray_shape)
            
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])  # [N_rays*N_samples, 96]

            # predict density
            new_encodings = self.density_net0(samples_enc)                  # [N_rays*N_samples, 256]
            new_encodings = torch.cat((new_encodings, samples_enc), -1)     # [N_rays*N_samples, 256+96]
            new_encodings = self.density_net1(new_encodings)                # [N_rays*N_samples, 256]
            raw_density = self.final_density(new_encodings).reshape((-1, num_samples, 1)) # [N_rays, N_samples, 1]
            
            # predict rgb
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(view_dirs.to(self.device))             # [N_rays, 27]
                viewdirs = torch.cat((viewdirs, view_dirs.to(self.device)), -1)          # [N_rays, 30]
                viewdirs = torch.tile(viewdirs[:, None, :], (1, num_samples, 1))    # [N_rays, N_samples, 30]
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))                    # [N_rays*N_samples, 30]
                new_encodings = self.rgb_net0(new_encodings)                             # [N_rays*N_samples, 256]
                new_encodings = torch.cat((new_encodings, viewdirs), -1)                 # [N_rays*N_samples, 30+256]
                new_encodings = self.rgb_net1(new_encodings)                             # [N_rays*N_samples, 286]
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, num_samples, 3))   # [N_rays, N_samples, 3]
            
            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals, rays_d.to(rgb.device), self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)


class NeRF(nn.Module):
    def __init__(self,
                 use_viewdirs=True,
                 randomized=False,
                 white_bkgd=True,
                 num_levels=2,
                 N_samples=64,
                 N_importance=128,
                 hidden=256,
                 density_noise=1,
                 resample_padding=0.01,
                 min_deg=0, max_deg=10,
                 viewdirs_min_deg=0, viewdirs_max_deg=4,
                 device=torch.device("cpu"),
                 return_raw=False
                 ):
        super(NeRF, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.density_input = 3 + (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.resample_padding = resample_padding
        self.hidden = hidden
        self.device = device
        self.return_raw = return_raw
        self.density_activation = nn.ReLU(True)

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),  # 63
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden), 
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        input_shape = hidden
        if self.use_viewdirs:
            input_shape = hidden // 2

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, input_shape),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)
        self.to(device)

    def forward(self, ray_batch):
        comp_rgbs = []
        distances = []
        accs = []

        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]                  # [N_rays, 3] each
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])                 # [N_rays, 1, 2]
        near, far = bounds[...,0], bounds[...,1]                             # [N_rays, 1]
        view_dirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None    # [N_rays, 3]
        
        for l in range(self.num_levels):
            # sample
            if l == 0:
                num_samples = self.N_samples
                t_vals, pts = sample_along_rays_nerf(rays_o, rays_d, self.N_samples,
                                                near, far, randomized=self.randomized, lindisp=False)
                # t_vals : [N_rays, N_samples]
                # pts : [N_rays, N_samples, 3]
            else:
                num_samples = self.N_samples + self.N_importance
                t_vals, pts = resample_along_rays_nerf(rays_o, rays_d, t_vals.to(rays_o.device),
                                                weights.to(rays_o.device), self.N_importance, randomized=self.randomized)
                # t_vals : [N_rays, N_samples+N_importance]
                # pts : [N_rays, N_importance+N_samples, 3]

            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(pts)                     # [N_rays, N_samples, 60]
            samples_enc = torch.cat([pts, samples_enc], -1)                 # [N_rays, N_samples, 63]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])  # [N_rays*N_samples, 63]
            
            # predict density
            new_encodings = self.density_net0(samples_enc)                  # [N_rays*N_samples, 256]
            new_encodings = torch.cat((new_encodings, samples_enc), -1)     # [N_rays*N_samples, 256+63]
            new_encodings = self.density_net1(new_encodings)                # [N_rays*N_samples, 256]
            raw_density = self.final_density(new_encodings).reshape((-1, num_samples, 1)) # [N_rays, N_samples, 1]
        
            # predict rgb
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(view_dirs.to(self.device))             # [N_rays, 24]
                viewdirs = torch.cat((viewdirs, view_dirs.to(self.device)), -1)          # [N_rays, 24+3]
                viewdirs = torch.tile(viewdirs[:, None, :], (1, num_samples, 1))         # [N_rays, N_samples, 24+3]
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))                    # [N_rays*N_samples, 24+3]
                new_encodings = self.rgb_net0(new_encodings)                             # [N_rays*N_samples, 256]
                new_encodings = torch.cat((new_encodings, viewdirs), -1)                 # [N_rays*N_samples, 27+256]
                new_encodings = self.rgb_net1(new_encodings)                             # [N_rays*N_samples, 128]
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, num_samples, 3))        # [N_rays, N_samples, 3]
            
            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb
            density = self.density_activation(raw_density)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering_nerf(rgb, density, t_vals, rays_d.to(rgb.device), self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
import torch
from model import NeRF

x = torch.rand((64, 11))
nerf = NeRF()

nerf(x)


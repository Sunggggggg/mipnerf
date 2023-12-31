from model import *

B, N = 64, 128
x = torch.rand((B, N, 3))
IPE = PositionalEncoding(0, 16)
print(IPE(x).shape)
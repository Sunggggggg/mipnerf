import torch
import torch.nn as nn

from .positional_encoding import get_2d_sincos_pos_embed

class ImageEmbed(nn.Module):
    """ 3D Image list to Image Embedding
    """
    def __init__(self, img_H=200, img_W=200, patch_H=50, patch_W=50, in_chans=3, embed_dim=256, norm_layer=None):
        super().__init__()
        num_patches_h = int(img_H // patch_H)
        num_patches_w = int(img_W // patch_W)
        num_patches = num_patches_h * num_patches_w

        self.num_patches = num_patches   # h*w
        self.embed_dim = embed_dim

        # embedding
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (1, patch_H, patch_W), stride= (1, patch_H, patch_W))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches*embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, num_patches_h, num_patches_w, False)
        
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().flatten().unsqueeze(0).unsqueeze(0)) 

        # init
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        """
        x       : input view tensors       [B, C, N, H, W]
        """
        B, C, N, H, W = x.shape

        x = self.proj(x)                            # [B, embed_dim, N, n, n]
        x = x.permute(0, 2, 3, 4, 1).flatten(2)     # [B, N, n, n, embed_dim] -> [B, N, n*n*embed_dim]
        x = self.norm(x)

        # Positional embedding
        pos_embed = self.pos_embed.expand(B, N, -1)
        x = x + pos_embed                          

        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_H=224, img_W=224, patch_H=16, patch_W=16, in_chans=3, embed_dim=256, norm_layer=None):
        super().__init__()
        
        self.num_patches = (img_H // patch_H) * (img_W // patch_W)   # h*w
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_H, patch_W), stride=(patch_H, patch_W))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        x       : input view tensors       [B, C, H, W]
        """
        B, C, H, W = x.shape

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
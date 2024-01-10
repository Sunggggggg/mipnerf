import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .model_vit import ImageEmbed, PatchEmbed
from .positional_encoding import matrix2angle, view_sinusoid_encoding, get_2d_sincos_pos_embed

print_parameters = lambda model : sum(p.numel() for p in model.parameters() if p.requires_grad)

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder
    """
    def __init__(self, img_H=400, img_W=400, patch_H=100, patch_W=100, in_chans=3,
                 embed_dim=512, depth=24, num_heads=16,
                 decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 emb_type='IMAGE', cam_pose_encoding=True):
        super().__init__()
        self.emb_type = emb_type
        self.cam_pose_encoding = cam_pose_encoding
        self.patch_H, self.patch_W = patch_H, patch_W
        self.num_patch_h, self.num_patch_w = int(img_H//patch_H), int(img_W//patch_W)
        # Embedding two type : (PatchEmbed / ImageEmbed)
        if emb_type == 'IMAGE' :
            # Encoder
            self.embed = ImageEmbed(img_H, img_W, patch_H, patch_W, in_chans, embed_dim)
            num_patches = self.embed.num_patches
            embed_dim = num_patches * embed_dim

            # Decoder
            decoder_embed_dim = num_patches * decoder_embed_dim
            self.decoder_pred = nn.Linear(decoder_embed_dim, img_H*img_W*in_chans, bias=True)            # decoder to images
        else :
            # Encoder
            self.embed = PatchEmbed(img_H, img_W, patch_H, patch_W, in_chans, embed_dim)
            num_patches = self.embed.num_patches

            # Decoder
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_H*patch_W*in_chans, bias=True)   # decoder to patches

        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def imagefiy(self, imgs):
        """input images to feature vectors (=patchify)

        imgs    : [B, 3, N, H, W]
        x       : [B, N, (HxWx3)]
        """
        x = imgs.permute(0, 2, 3, 4, 1)     # [B, N, H, W, 3]
        x = x.flatten(2)                    # [B, N, H*W*3]

        return x
    
    def patchify(self, imgs):
        """
        imgs: (B, 3, H*n, W*n)
        x: (B, n*n, patch_size**2 *3)
        """
        p_h, p_w = self.patch_H, self.patch_W
        h, w = self.num_patch_h, self.num_patch_w
        assert h == w, "num_patch h, w same!"

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p_h, w, p_w))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h*p_w* 3))
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device) 
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)       # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, poses, mask_ratio=0.75):
        """
        x       : input view tensors       [B, C, N, H, W]        
        poses   : input view poses         [B, N, 4, 4]
        """
        # embedding
        x = self.embed(x)     #[B, N, embed_dim]
        
        # Positional encoding
        if self.cam_pose_encoding :
            theta, phi = matrix2angle(poses)
            pos_embed = view_sinusoid_encoding(theta, phi, x.shape[-1], True).to(x.device)
        else :
            B, N, D =x.shape
            n = int(N**.5)
            pos_embed = get_2d_sincos_pos_embed(x.shape[-1], n, n, cls_token=True)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)            # [1, N+1, embed_dim]
            pos_embed = pos_embed.type(x.type())
        
        x = x + pos_embed[:, 1:, :]                                                 # [B, N, embed_dim]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_decoder(self, x, poses, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        if self.cam_pose_encoding :
            theta, phi = matrix2angle(poses)
            decoder_pos_embed = view_sinusoid_encoding(theta, phi, x.shape[-1], True).to(x.device)
        else :
            B, N, D =x.shape
            n = int(N**.5)  
            decoder_pos_embed = get_2d_sincos_pos_embed(x.shape[-1], n, n, cls_token=True)
            decoder_pos_embed = torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)            # [1, N+1, embed_dim]
            decoder_pos_embed = decoder_pos_embed.type(x.type())

        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
    
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [B, 3, N, H, W]
        pred: [B, N, H*W*3]
        mask: [B, L], 0 is keep, 1 is remove, 
        """
        B, N, D = pred.shape

        if self.emb_type == "IMAGE" :
            target = self.imagefiy(imgs)
        else :
            target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, poses=None, mask_ratio=0.75):
        """
        imgs        [B, 3, N, H, W]
        poses       [B, N, 4, 4] 
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, poses, mask_ratio)
        pred = self.forward_decoder(latent, poses, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask

class OnlyEncoder(nn.Module):
    def __init__(self, img_H=400, img_W=400, patch_H=100, patch_W=100, in_chans=3,
                embed_dim=512, depth=24, num_heads=16,
                decoder_embed_dim=256, mlp_ratio=4., norm_layer=nn.LayerNorm, 
                emb_type='IMAGE', cam_pose_encoding=True):
        super().__init__()
        self.emb_type = emb_type
        self.cam_pose_encoding = cam_pose_encoding
        self.patch_H, self.patch_W = patch_H, patch_W
        self.num_patch_h, self.num_patch_w = int(img_H//patch_H), int(img_W//patch_W)
        # Embedding two type : (PatchEmbed / ImageEmbed)
        if emb_type == 'IMAGE' :
            # Encoder
            self.embed = ImageEmbed(img_H, img_W, patch_H, patch_W, in_chans, embed_dim)
            num_patches = self.embed.num_patches
            embed_dim = num_patches * embed_dim
            
            # Decoder
            decoder_embed_dim = num_patches * decoder_embed_dim
        else :
            # Encoder
            self.embed = PatchEmbed(img_H, img_W, patch_H, patch_W, in_chans, embed_dim)
            num_patches = self.embed.num_patches

        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
    def forward(self, imgs, poses, N_inputs, N_fewshots):
        x = self.embed(imgs)     #[B, N, n*n*embed_dim]
        
        if self.cam_pose_encoding :
            theta, phi = matrix2angle(poses)
            pos_embed = view_sinusoid_encoding(theta, phi, x.shape[-1], True).to(x.device)
        else :
            B, N, D =x.shape
            n = int(N**.5)
            pos_embed = get_2d_sincos_pos_embed(x.shape[-1], n, n, cls_token=True)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)            # [1, N+1, embed_dim]
            pos_embed = pos_embed.type(x.type())

        x = x + pos_embed[:, 1:, :]         #[B, N, n*n*embed_dim]

        x = x[:, :N_fewshots, :]            #[B, F, n*n*embed_dim]    

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                   #[B, F+1, n*n*embed_dim]

        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], N_inputs - N_fewshots, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x = torch.cat([x[:, :1, :], x_], dim=1) 

        if self.cam_pose_encoding :
            theta, phi = matrix2angle(poses)
            decoder_pos_embed = view_sinusoid_encoding(theta, phi, x.shape[-1], True).to(x.device)
        else :
            B, N, D =x.shape
            n = int(N**.5)
            decoder_pos_embed = get_2d_sincos_pos_embed(x.shape[-1], n, n, cls_token=True)
            decoder_pos_embed = torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)            # [1, N+1, embed_dim]
            decoder_pos_embed = decoder_pos_embed.type(x.type())

        x = x + decoder_pos_embed

        return x

def mae_patch_embedding(args, H, W, n=5):
    """ Use patch embedding for feature extracting
    """
    model = MaskedAutoencoderViT(img_H=H*n, img_W=W*n, patch_H=H, patch_W=W, embed_dim=args.embed_dim, 
                                 depth=args.depth, num_heads=args.num_heads, decoder_embed_dim=args.decoder_embed_dim,
                                 decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, mlp_ratio=4, 
                                 norm_layer=nn.LayerNorm, norm_pix_loss=args.norm_pix_loss, 
                                 emb_type=args.emb_type, cam_pose_encoding=args.cam_pose_encoding)
    
    print("Build patch embedding MAE :", print_parameters(model))
    print("Summary")
    print(f"Image size : ({H*n}, {W*n})")
    print(f"Patch size : ({H}, {W})")
    print(f"Embed_dim (E/D) : {args.embed_dim}, {args.decoder_embed_dim}")

    return model

def mae_image_embedding(args, H, W, n=5):
    """ Use image embedding for feature extracting
    """
    k = args.image_token**0.5
    patch_H, patch_W = int(H//k), int(W//k)

    model = MaskedAutoencoderViT(img_H=H, img_W=W, patch_H=patch_H, patch_W=patch_W, embed_dim=args.embed_dim, 
                                 depth=args.depth, num_heads=args.num_heads, decoder_embed_dim=args.decoder_embed_dim,
                                 decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, mlp_ratio=4, 
                                 norm_layer=nn.LayerNorm, norm_pix_loss=args.norm_pix_loss, 
                                 emb_type=args.emb_type, cam_pose_encoding=args.cam_pose_encoding)
    print("Build image embedding MAE :", print_parameters(model))
    print("Summary")
    print(f"Image size : ({H}, {W})")
    print(f"Patch size : ({patch_H}, {patch_W})")
    print(f"Embed_dim (E/D) : {args.embed_dim*args.image_token}, {args.decoder_embed_dim*args.image_token}")

    return model

def encoder_image_embedding(args, H, W):
    k = args.image_token**0.5
    patch_H, patch_W = int(H//k), int(W//k)

    encoder = OnlyEncoder(img_H=H, img_W=W, patch_H=patch_H, patch_W=patch_W, 
                          embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                          decoder_embed_dim=args.decoder_embed_dim, mlp_ratio=4, norm_layer=nn.LayerNorm, 
                          emb_type=args.emb_type, cam_pose_encoding=args.cam_pose_encoding)
    
    print("Load MipNeRF model weight :", args.mae_weight)
    print("Build image embedding MAE :", print_parameters(encoder))
    print(f"Image size : ({H}, {W})")
    print(f"Patch size : ({patch_H}, {patch_W})")
    print(f"Embed_dim (E/D) : {args.embed_dim*args.image_token}, {args.decoder_embed_dim*args.image_token}")

    return encoder

def encoder_patch_embedding(args, H, W):
    n = int(args.mae_input**0.5)
    
    encoder = OnlyEncoder(img_H=H*n, img_W=W*n, patch_H=H, patch_W=W, 
                          embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, 
                          decoder_embed_dim=args.decoder_embed_dim, mlp_ratio=4, norm_layer=nn.LayerNorm, 
                          emb_type=args.emb_type, cam_pose_encoding=args.cam_pose_encoding)
    
    print("Load MipNeRF model weight :", args.mae_weight)
    print("Build image embedding MAE :", print_parameters(encoder))
    print(f"Image size : ({H*n}, {W*n})")
    print(f"Patch size : ({H}, {W})")
    print(f"Embed_dim (E/D) : {args.embed_dim}, {args.decoder_embed_dim}")

    return encoder

IMAGE = encoder_image_embedding
PATCH = encoder_patch_embedding
IMAGE_MAE = mae_image_embedding
PATCH_MAE = mae_patch_embedding

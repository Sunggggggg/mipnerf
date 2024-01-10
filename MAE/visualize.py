import os
import torch
import numpy as np
import matplotlib.pyplot as plt

tensor2img = lambda tensor : tensor.detach().cpu().numpy()
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def image_plot(images, emb_type='IMAGE', row=5, save_fig=None):
    """ 
    images (Tensor)         : [3, N, H, W] 
    images (numpy)          : [3, N, H, W] 
    """
    if type(images) == torch.Tensor :
        images = tensor2img(images.permute(1, 2, 3, 0)) # numpy [N, H, W, 3]
    else :
        images = images.transpose(1, 2, 3, 0)           # numpy [N, H, W, 3]
    
    
    N = images.shape[0]
    col = int(N / row)
    assert N == row*col, f"Check row and col {row}, {col}"

    # Stack images
    vis_images = np.vstack([np.hstack(images[i*col:(i+1)*col]) for i in range(row)])

    # 
    if save_fig is not None :
        vis_images = to8b(vis_images)
        
        plt.imshow(vis_images)
        plt.axis('off')
        plt.savefig(save_fig)
        plt.close()

    return vis_images
import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips import LPIPS

def get_metric(rgbs, targets, lpips=None, device=torch.device('cuda')):
    """
    rgbs, targets : [N, H, W, 3]
    """
    lpips_vgg = LPIPS(net="vgg").to(device)

    rgbs = np.clip(rgbs, 0., 1.).astype(targets.dtype)
    assert targets.dtype == rgbs.dtype, "Different data type"
    
    test_view = targets.shape[0]
    psnr = np.mean([peak_signal_noise_ratio(targets[i],rgbs[i]) for i in range(test_view)])
    ssim = np.mean([structural_similarity(targets[i], rgbs[i], data_range=rgbs.max()-rgbs.min(), channel_axis=-1) for i in range(test_view)])

    lpips_score = torch.mean(lpips_vgg(torch.from_numpy(targets).permute(0, -1, 1, 2).type(torch.FloatTensor).to(device), 
                                        torch.from_numpy(rgbs).permute(0, -1, 1, 2).type(torch.FloatTensor).to(device)), dim=0).item()
    
    return psnr, ssim, lpips_score
    
    
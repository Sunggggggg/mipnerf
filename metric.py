import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips import LPIPS

def get_metric(rgbs, targets, lpips=None, device=torch.device('cuda')):
    rgbs = rgbs.astype(targets.dtype)
    assert targets.dtype == rgbs.dtype, "Different data type"
    
    test_view = targets.shape[0]
    psnr = np.mean([peak_signal_noise_ratio(rgbs[i], targets[i]) for i in range(test_view)])
    ssim = np.mean([structural_similarity(rgbs[i], targets[i], data_range=rgbs.max()-rgbs.min(), channel_axis=-1) for i in range(test_view)])

    if lpips is not None :
        lpips_vgg = LPIPS(net="vgg").cuda() 
        lpips_score = torch.mean([lpips_vgg(torch.from_numpy(targets[i]).cuda().permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor).to(device), 
                                torch.from_numpy(rgbs[i]).cuda().permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor).to(device)) for i in range(test_view)]).item()
        return psnr, ssim, lpips_score
    
    else: 
        return psnr, ssim, torch.tensor([0])
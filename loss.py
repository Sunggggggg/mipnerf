import torch

class MipNeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, coarse_weight_decay=0.1):
        super(MipNeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay

    def forward(self, input, target, mask):
        """
        Args
        input       : [2, B, 3]  coarse+fine = 2
        target      : [B, 3]
        mask        : [B, 1]

        Return
        loss        : Scalar
        psnrs       : [Coarse psnr, Fine psnr]
        """
        losses = []
        psnrs = []
        for rgb in input:
            mse = (mask * ((rgb - target[..., :3]) ** 2)).sum() / mask.sum()
            losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = torch.stack(losses)
        loss = self.coarse_weight_decay * torch.sum(losses[:-1]) + losses[-1]
        return loss, losses, torch.Tensor(psnrs)

class NeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(NeRFLoss, self).__init__()

    def forward(self, input, target):
        """
        Args
        input       : [2, B, 3]  coarse+fine = 2
        target      : [B, 3]

        Return
        loss        : Scalar
        psnrs       : [Coarse psnr, Fine psnr]
        """
        losses = []
        psnrs = []
        for rgb in input:
            mse = ((rgb - target[..., :3]) ** 2).sum()
            losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = torch.stack(losses)            # [2, 1]
        loss = torch.sum(losses)
        return loss, losses, torch.Tensor(psnrs)


def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)

class MAELoss(torch.nn.modules.loss._Loss):
    def __init__(self, loss_type="COSINE"):
        super(MAELoss, self).__init__()
        self.loss_type = loss_type
        self.cossim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, gt_feat, object_feat):
        """ Get loss, angle btw gt vector and object vector  
        gt_feat, object_feat : [1, N, D]
        """
        if self.loss_type == "COSINE":
            loss = self.cossim(gt_feat, object_feat).mean()
        else : # "PERCE"
            pass
        
        return loss
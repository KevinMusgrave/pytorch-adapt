import torch
import torch.nn.functional as F


# https://openaccess.thecvf.com/content_iccv_2015/papers/Tzeng_Simultaneous_Deep_Transfer_ICCV_2015_paper.pdf
class UniformDistributionLoss(torch.nn.Module):
    # *args to make it work as a drop in replacement for CrossEntropyLoss
    def forward(self, x, *args):
        probs = F.log_softmax(x, dim=1)
        avg_probs = torch.mean(probs, dim=1)
        return -torch.mean(avg_probs)

import torch


class GrayscaleToRGB:
    def __call__(self, x):
        return torch.cat([x, x, x], dim=0)

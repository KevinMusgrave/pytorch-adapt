import torch
import torchvision.transforms as T

from .constants import IMAGENET_MEAN, IMAGENET_STD


class GrayscaleToRGB:
    def __call__(self, x):
        return torch.cat([x, x, x], dim=0)


def get_mnist_transform(domain, *_):
    if domain == "mnist":
        return T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                GrayscaleToRGB(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif domain == "mnistm":
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def get_resnet_transform(domain, train, is_training):
    transform = [T.Resize(256)]
    if is_training:
        transform += [
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
        ]
    else:
        transform += [T.CenterCrop(224)]

    transform += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transform)

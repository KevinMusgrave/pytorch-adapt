import torch
import torchvision.transforms as T

from .constants import IMAGENET_MEAN, IMAGENET_STD


class GrayscaleToRGB:
    def __call__(self, x):
        return torch.cat([x, x, x], dim=0)


def get_mnist_transform(domain, **kwargs):
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


def get_resnet_transform(is_training, **kwargs):
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


def get_timm_transform(is_training, **kwargs):
    from timm.data.transforms_factory import create_transform

    return create_transform(
        input_size=224, is_training=is_training, auto_augment="original"
    )

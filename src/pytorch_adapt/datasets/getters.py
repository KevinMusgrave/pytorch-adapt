import torchvision.transforms as T
from torchvision import datasets

from ..utils import common_functions as c_f
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD
from ..utils.transforms import GrayscaleToRGB
from .combined_source_and_target import CombinedSourceAndTargetDataset
from .concat_dataset import ConcatDataset
from .mnistm import MNISTM
from .office31 import Office31
from .officehome import OfficeHome
from .source_dataset import SourceDataset
from .target_dataset import TargetDataset


def get_multiple(dataset_getter, domains, *args):
    return ConcatDataset([dataset_getter(d, *args) for d in domains])


def get_datasets(
    dataset_getter,
    src_domains,
    target_domains,
    folder,
    download=False,
    return_target_with_labels=False,
    transform_getter=None,
):
    def getter(domains, train, is_training):
        return get_multiple(
            dataset_getter,
            domains,
            train,
            is_training,
            folder,
            download,
            transform_getter,
        )

    output = {}
    output["src_train"] = SourceDataset(getter(src_domains, True, False))
    output["src_val"] = SourceDataset(getter(src_domains, False, False))
    output["target_train"] = TargetDataset(getter(target_domains, True, False))
    output["target_val"] = TargetDataset(getter(target_domains, False, False))
    if return_target_with_labels:
        output["target_train_with_labels"] = SourceDataset(
            getter(target_domains, True, False), domain=1
        )
        output["target_val_with_labels"] = SourceDataset(
            getter(target_domains, False, False), domain=1
        )
    output["train"] = CombinedSourceAndTargetDataset(
        SourceDataset(getter(src_domains, True, True)),
        TargetDataset(getter(target_domains, True, True)),
    )
    return output


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


def _get_mnist_mnistm(domain, train, is_training, folder, download, transform_getter):
    transform_getter = c_f.default(transform_getter, get_mnist_transform)
    transform = transform_getter(domain, train, is_training)
    if domain == "mnist":
        return datasets.MNIST(
            folder, train=train, transform=transform, download=download
        )
    elif domain == "mnistm":
        return MNISTM(folder, train, transform, download=download)


def get_mnist_mnistm(*args, **kwargs):
    return get_datasets(_get_mnist_mnistm, *args, **kwargs)


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


def standard_dataset(cls):
    def fn(domain, train, is_training, folder, download, transform_getter):
        transform_getter = c_f.default(transform_getter, get_resnet_transform)
        transform = transform_getter(domain, train, is_training)
        return cls(
            root=folder,
            domain=domain,
            train=train,
            transform=transform,
            download=download,
        )

    return fn


def get_office31(*args, **kwargs):
    return get_datasets(standard_dataset(Office31), *args, **kwargs)


def get_officehome(*args, **kwargs):
    return get_datasets(standard_dataset(OfficeHome), *args, **kwargs)

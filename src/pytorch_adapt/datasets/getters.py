import torchvision.transforms as T
from torchvision import datasets

from ..utils import common_functions as c_f
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD
from ..utils.transforms import GrayscaleToRGB
from .combined_source_and_target import CombinedSourceAndTargetDataset
from .concat_dataset import ConcatDataset
from .mnistm import MNISTM
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


def get_mnist(domain, train, is_training, folder, download, transform_getter):
    get_transform = c_f.default(transform_getter, get_mnist_transform)
    transform = get_mnist_transform(domain)
    if domain == "mnist":
        return datasets.MNIST(
            folder, train=train, download=download, transform=transform
        )
    elif domain == "mnistm":
        return MNISTM(folder, train, transform, download=download)


def get_mnist_mnistm(*args, **kwargs):
    return get_datasets(get_mnist, *args, **kwargs)

import torchvision.transforms as T
from torchvision import datasets

from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD
from ..utils.transforms import GrayscaleToRGB
from .combined_source_and_target import CombinedSourceAndTargetDataset
from .concat_dataset import ConcatDataset
from .mnistm import MNISTM
from .source_dataset import SourceDataset
from .target_dataset import TargetDataset


def get_multiple(domains, train, is_training, folder, dataset_getter, download):
    return ConcatDataset(
        [dataset_getter(d, train, is_training, folder, download) for d in domains]
    )


def get_datasets(
    src_domains,
    target_domains,
    folder,
    dataset_getter,
    download,
    return_target_with_labels,
):
    def getter(domains, train, is_training):
        return get_multiple(
            domains, train, is_training, folder, dataset_getter, download
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


def get_mnist(domain, train, is_training, folder, download):
    if domain == "mnist":
        transform = T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                GrayscaleToRGB(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return datasets.MNIST(
            folder, train=train, download=download, transform=transform
        )
    elif domain == "mnistm":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return MNISTM(folder, train, transform, download=download)


def get_mnist_mnistm(
    src_domains, target_domains, folder, download=False, return_target_with_labels=False
):
    return get_datasets(
        src_domains,
        target_domains,
        folder,
        get_mnist,
        download,
        return_target_with_labels,
    )

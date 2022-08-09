from torchvision import datasets

from ..transforms.classification import get_mnist_transform, get_resnet_transform
from ..utils import common_functions as c_f
from .combined_source_and_target import CombinedSourceAndTargetDataset
from .concat_dataset import ConcatDataset
from .domainnet import DomainNet126
from .mnistm import MNISTM
from .office31 import Office31
from .officehome import OfficeHome
from .source_dataset import SourceDataset
from .target_dataset import TargetDataset


def get_multiple(dataset_getter, domains, **kwargs):
    return ConcatDataset([dataset_getter(domain=d, **kwargs) for d in domains])


def get_datasets(
    dataset_getter,
    src_domains,
    target_domains,
    folder,
    download=False,
    return_target_with_labels=False,
    supervised=False,
    transform_getter=None,
):
    def getter(domains, train, is_training):
        return get_multiple(
            dataset_getter,
            domains,
            train=train,
            is_training=is_training,
            root=folder,
            download=download,
            transform_getter=transform_getter,
        )

    if not src_domains and not target_domains:
        raise ValueError(
            "At least one of src_domains and target_domains must be provided"
        )

    output = {}
    if src_domains:
        output["src_train"] = SourceDataset(getter(src_domains, True, False))
        output["src_val"] = SourceDataset(getter(src_domains, False, False))
    if target_domains:
        output["target_train"] = TargetDataset(
            getter(target_domains, True, False), supervised=supervised
        )
        output["target_val"] = TargetDataset(
            getter(target_domains, False, False), supervised=supervised
        )
        # For academic setting: unsupervised learning w/ seperate target datasets that have gt lables for eval.
        if return_target_with_labels:
            output["target_train_with_labels"] = TargetDataset(
                getter(target_domains, True, False), domain=1, supervised=True
            )
            output["target_val_with_labels"] = TargetDataset(
                getter(target_domains, False, False), domain=1, supervised=True
            )
    if src_domains and target_domains:
        output["train"] = CombinedSourceAndTargetDataset(
            SourceDataset(getter(src_domains, True, True)),
            TargetDataset(getter(target_domains, True, True)),
        )
    elif src_domains:
        output["train"] = SourceDataset(getter(src_domains, True, True))
    elif target_domains:
        output["train"] = TargetDataset(getter(target_domains, True, True))
    return output


def _get_mnist_mnistm(is_training, transform_getter, **kwargs):
    transform_getter = c_f.default(transform_getter, get_mnist_transform)
    domain = kwargs["domain"]
    kwargs["transform"] = transform_getter(domain, kwargs["train"], is_training)
    kwargs.pop("domain")
    if domain == "mnist":
        return datasets.MNIST(**kwargs)
    elif domain == "mnistm":
        return MNISTM(**kwargs)


def get_mnist_mnistm(*args, **kwargs):
    return get_datasets(_get_mnist_mnistm, *args, **kwargs)


def standard_dataset(cls):
    def fn(is_training, transform_getter, **kwargs):
        transform_getter = c_f.default(transform_getter, get_resnet_transform)
        kwargs["transform"] = transform_getter(
            kwargs["domain"], kwargs["train"], is_training
        )
        return cls(**kwargs)

    return fn


def get_office31(*args, **kwargs):
    return get_datasets(standard_dataset(Office31), *args, **kwargs)


def get_officehome(*args, **kwargs):
    return get_datasets(standard_dataset(OfficeHome), *args, **kwargs)


def get_domainnet126(*args, **kwargs):
    return get_datasets(standard_dataset(DomainNet126), *args, **kwargs)

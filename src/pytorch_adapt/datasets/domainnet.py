import os
from collections import OrderedDict

from .base_dataset import BaseDataset
from .utils import check_img_paths, check_length


class DomainNet(BaseDataset):
    """
    A large dataset used in "Moment Matching for Multi-Source Domain Adaptation".
    It consists of 345 classes in 6 domains:
    clipart, infograph, painting, quickdraw, real, sketch
    """

    def __init__(self, root: str, domain: str, train: bool, transform, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 6 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "domainnet", f"{domain}_{name}.txt")
        img_dir = os.path.join(root, "domainnet")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, domain)
        check_length(
            self,
            {
                "clipart": {"train": 33525, "test": 14604}[name],
                "infograph": {"train": 36023, "test": 15582}[name],
                "painting": {"train": 50416, "test": 21850}[name],
                "quickdraw": {"train": 120750, "test": 51750}[name],
                "real": {"train": 120906, "test": 52041}[name],
                "sketch": {"train": 48212, "test": 20916}[name],
            }[domain],
        )
        self.labels = [int(x[1]) for x in content]
        self.transform = transform


class DomainNet126Full(BaseDataset):
    """
    A subset of DomainNet consisting of 126 classes and 4 domains:
    clipart, painting, real, sketch
    """

    def __init__(self, root: str, domain: str, transform, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 4 domains
            transform: The image transform applied to each sample.
        """
        super().__init__(domain=domain, **kwargs)
        filenames = [
            f"labeled_source_images_{domain}",
            f"labeled_target_images_{domain}_1",
            f"labeled_target_images_{domain}_3",
            f"unlabeled_target_images_{domain}_1",
            f"unlabeled_target_images_{domain}_3",
            f"validation_target_images_{domain}_3",
        ]
        filenames = [os.path.join(root, "domainnet", f"{f}.txt") for f in filenames]
        img_dir = os.path.join(root, "domainnet")

        content = OrderedDict()
        for f in filenames:
            with open(f) as fff:
                for line in fff:
                    path, label = line.rstrip().split(" ")
                    content[path] = label

        self.img_paths = [os.path.join(img_dir, x) for x in content.keys()]
        check_img_paths(img_dir, self.img_paths, domain)
        self.labels = [int(x) for x in content.values()]
        self.transform = transform


class DomainNet126(BaseDataset):
    """
    A custom train/test split of DomainNet126Full.
    """

    def __init__(self, root: str, domain: str, train: bool, transform, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 4 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "domainnet", f"{domain}126_{name}.txt")
        img_dir = os.path.join(root, "domainnet")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, domain)
        check_length(
            self,
            {
                "clipart": {"train": 14962, "test": 3741}[name],
                "painting": {"train": 25201, "test": 6301}[name],
                "real": {"train": 56286, "test": 14072}[name],
                "sketch": {"train": 19665, "test": 4917}[name],
            }[domain],
        )
        self.labels = [int(x[1]) for x in content]
        self.transform = transform

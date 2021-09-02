import os

import torch
from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset
from .utils import check_img_paths, check_length


class Office31Full(BaseDataset):
    """
    A small dataset consisting of 31 classes in 3 domains:
    amazon, dslr, webcam.
    """

    def __init__(self, root: str, domain: str, transform):
        """
        Arguments:
            root: The dataset must be located at ```<root>/office31```
            domain: One of the 3 domains
            transform: The image transform applied to each sample.
        """

        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "office31", domain, "images"), transform=self.transform
        )
        check_length(self, {"amazon": 2817, "dslr": 498, "webcam": 795}[domain])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class Office31(BaseDataset):
    """
    A custom train/test split of Office31Full.
    """

    def __init__(self, root: str, domain: str, train: bool, transform, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/office31```
            domain: One of the 3 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """

        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "office31", f"{domain}_{name}.txt")
        img_dir = os.path.join(root, "office31")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, domain)
        check_length(
            self,
            {
                "amazon": {"train": 2253, "test": 564}[name],
                "dslr": {"train": 398, "test": 100}[name],
                "webcam": {"train": 636, "test": 159}[name],
            }[domain],
        )
        self.labels = [int(x[1]) for x in content]
        self.transform = transform

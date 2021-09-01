import os

import torch
from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset


class Office31Full(BaseDataset):
    def __init__(self, root, domain, transform):
        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "office31", domain, "images"), transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class Office31(BaseDataset):
    def __init__(self, root, domain, train, transform, **kwargs):
        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "office31", f"{domain}_{name}.txt")
        img_dir = os.path.join(root, "office31")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        self.labels = [int(x[1]) for x in content]
        self.transform = transform

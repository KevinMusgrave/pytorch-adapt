import os
from collections import OrderedDict

from .base_dataset import BaseDataset


class DomainNet(BaseDataset):
    def __init__(self, root, domain, train, transform, **kwargs):
        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "domainnet", f"{domain}_{name}.txt")
        img_dir = os.path.join(root, "domainnet")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        self.labels = [int(x[1]) for x in content]
        self.transform = transform


class DomainNet126Full(BaseDataset):
    def __init__(self, root, domain, transform, **kwargs):
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
        self.labels = [int(x) for x in content.values()]
        self.transform = transform


class DomainNet126(BaseDataset):
    def __init__(self, root, domain, train, transform, **kwargs):
        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "domainnet", f"{domain}126_{name}.txt")
        img_dir = os.path.join(root, "domainnet")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        self.labels = [int(x[1]) for x in content]
        self.transform = transform

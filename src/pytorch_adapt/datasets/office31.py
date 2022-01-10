import os

from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_train


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


class Office31(BaseDownloadableDataset):
    """
    A custom train/test split of Office31Full.
    """

    url = "https://cornell.box.com/shared/static/3v2ftdkdhpz1lbbr4uhu0135w7m79p7q"
    filename = "office31.tar.gz"
    md5 = "89818e596f3cdda1d56da0f077435faa"

    def __init__(self, root: str, domain: str, train: bool, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/office31```
            domain: One of the 3 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.train = check_train(train)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform

    def set_paths_and_labels(self, root):
        name = "train" if self.train else "test"
        labels_file = os.path.join(root, "office31", f"{self.domain}_{name}.txt")
        img_dir = os.path.join(root, "office31")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                "amazon": {"train": 2253, "test": 564}[name],
                "dslr": {"train": 398, "test": 100}[name],
                "webcam": {"train": 636, "test": 159}[name],
            }[self.domain],
        )
        self.labels = [int(x[1]) for x in content]

import os

from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_train


class OfficeHomeFull(BaseDataset):
    def __init__(self, root: str, domain: str, transform):
        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "officehome", domain),
            transform=self.transform,
        )
        check_length(
            self, {"art": 2427, "clipart": 4365, "product": 4439, "real": 4357}[domain]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class OfficeHome(BaseDownloadableDataset):
    url = "https://cornell.box.com/shared/static/xwsbubtcr8flqfuds5f6okqbr3z0w82t"
    filename = "officehome_resized.tar.gz"
    md5 = "52d6039512434aa561d66de9c10828c3"

    def __init__(self, root: str, domain: str, train: bool, transform=None, **kwargs):
        self.train = check_train(train)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform

    def set_paths_and_labels(self, root):
        name = "train" if self.train else "test"
        labels_file = os.path.join(root, "officehome", f"{self.domain}_{name}.txt")
        img_dir = os.path.join(root, "officehome")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                "art": {"train": 1941, "test": 486}[name],
                "clipart": {"train": 3492, "test": 873}[name],
                "product": {"train": 3551, "test": 888}[name],
                "real": {"train": 3485, "test": 872}[name],
            }[self.domain],
        )
        self.labels = [int(x[1]) for x in content]

import os
import tarfile

from torchvision.datasets.utils import download_url

from ..utils import common_functions as c_f
from .base_dataset import BaseDataset
from .utils import check_length


class MNISTM(BaseDataset):
    """
    The dataset used in "Domain-Adversarial Training of Neural Networks".
    It consists of colored MNIST digits.
    """

    url = "https://cornell.box.com/shared/static/jado7quprg6hzzdubvwzh9umr75damwi"
    filename = "mnist_m.tar.gz"
    md5 = "859df31c91afe82e80e5012ba928f279"

    def __init__(
        self, root: str, train: bool, transform=None, download=False, **kwargs
    ):
        """
        Arguments:
            root: The dataset must be located at ```<root>/mnist_m```
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__(domain="MNISTM", **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        if download:
            try:
                self.set_paths_and_labels(root, train, transform)
            except (FileNotFoundError, ValueError):
                self.download_dataset(root)
                self.set_paths_and_labels(root, train, transform)
        else:
            self.set_paths_and_labels(root, train, transform)

    def set_paths_and_labels(self, root, train, transform):
        name = "train" if train else "test"
        labels_file = os.path.join(root, "mnist_m", f"mnist_m_{name}_labels.txt")
        img_dir = os.path.join(root, "mnist_m", f"mnist_m_{name}")
        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_length(self, {"train": 59001, "test": 9001}[name])
        self.labels = [int(x[1]) for x in content]
        self.transform = transform

    def download_dataset(self, root):
        download_url(self.url, root, filename=self.filename, md5=self.md5)
        with tarfile.open(os.path.join(root, self.filename), "r:gz") as tar:
            tar.extractall(path=root, members=c_f.extract_progress(tar))

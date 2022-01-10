import os

from .base_dataset import BaseDownloadableDataset
from .utils import check_length, check_train


class MNISTM(BaseDownloadableDataset):
    """
    The dataset used in "Domain-Adversarial Training of Neural Networks".
    It consists of colored MNIST digits.
    """

    url = "https://cornell.box.com/shared/static/jado7quprg6hzzdubvwzh9umr75damwi"
    filename = "mnist_m.tar.gz"
    md5 = "859df31c91afe82e80e5012ba928f279"

    def __init__(self, root: str, train: bool, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/mnist_m```
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.train = check_train(train)
        super().__init__(root=root, domain="MNISTM", **kwargs)
        self.transform = transform

    def set_paths_and_labels(self, root):
        name = "train" if self.train else "test"
        labels_file = os.path.join(root, "mnist_m", f"mnist_m_{name}_labels.txt")
        img_dir = os.path.join(root, "mnist_m", f"mnist_m_{name}")
        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_length(self, {"train": 59001, "test": 9001}[name])
        self.labels = [int(x[1]) for x in content]

import os

from .base_dataset import BaseDataset
from .utils import check_length


class MNISTM(BaseDataset):
    """
    The dataset used in "Domain-Adversarial Training of Neural Networks".
    It consists of colored MNIST digits.
    """

    def __init__(self, root: str, train: bool, transform, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/mnist_m```
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__(domain="MNISTM", **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "mnist_m", f"mnist_m_{name}_labels.txt")
        img_dir = os.path.join(root, "mnist_m", f"mnist_m_{name}")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_length(self, {"train": 59001, "test": 9001}[name])
        self.labels = [int(x[1]) for x in content]
        self.transform = transform

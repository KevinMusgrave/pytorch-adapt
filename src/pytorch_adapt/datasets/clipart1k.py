import os

from torchvision.datasets import VisionDataset, VOCDetection

from .utils import check_length, maybe_download
from .voc_multilabel import process_voc_style_dataset


class Clipart1kMultiLabel(VOCDetection):
    url = "http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/clipart.zip"
    filename = "clipart1k.zip"
    md5 = "883e2c03eaff39e17ea60d43a3899224"

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        download: bool = False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        VisionDataset.__init__(self, root, transforms, transform, target_transform)
        dataset_root = os.path.join(self.root, "clipart1k")

        def rename_fn():
            os.rename(os.path.join(self.root, "clipart"), dataset_root)

        maybe_download(
            process_voc_style_dataset,
            {
                "cls": self,
                "dataset_root": dataset_root,
                "image_set": image_set,
                "download": download,
                "rename_fn": rename_fn,
                "exclude_list": [
                    "681633840"
                ],  # https://github.com/naoto0804/cross-domain-detection/issues/39
            },
        )

        check_length(self, {"train": 499, "test": 500}[image_set])

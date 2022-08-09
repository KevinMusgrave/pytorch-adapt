import os

import torch
from torchvision.datasets import VOCDetection
from torchvision.datasets.utils import download_and_extract_archive

from .utils import maybe_download


# modified from https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py
def process_voc_style_dataset(cls, dataset_root, image_set, download, rename_fn=None):
    if download:
        download_and_extract_archive(
            cls.url, cls.root, filename=cls.filename, md5=cls.md5
        )
        if rename_fn:
            rename_fn()

    splits_dir = os.path.join(dataset_root, "ImageSets", cls._SPLITS_DIR)
    split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
    with open(os.path.join(split_f)) as f:
        file_names = [x.strip() for x in f.readlines()]

    image_dir = os.path.join(dataset_root, "JPEGImages")
    cls.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

    target_dir = os.path.join(dataset_root, cls._TARGET_DIR)
    cls.targets = [
        os.path.join(target_dir, x + cls._TARGET_FILE_EXT) for x in file_names
    ]

    assert len(cls.images) == len(cls.targets)


def get_multilabel(img, info, CLASSNAME_TO_IDX, NUM_CLASSES):
    classes = [CLASSNAME_TO_IDX[x["name"]] for x in info["annotation"]["object"]]
    label = torch.zeros(NUM_CLASSES, dtype=int)
    label[classes] = 1
    return img, label


class VOCMultiLabel(VOCDetection):
    NUM_CLASSES = 20
    CLASSNAMES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    CLASSNAME_TO_IDX = {x: i for i, x in enumerate(CLASSNAMES)}

    def __init__(self, **kwargs):
        maybe_download(super().__init__, kwargs)

    def __getitem__(self, idx):
        img, info = super().__getitem__(idx)
        return get_multilabel(img, info, self.CLASSNAME_TO_IDX, self.NUM_CLASSES)

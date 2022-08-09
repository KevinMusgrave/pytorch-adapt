import os

import torch
from torchvision.datasets import VOCDetection
from torchvision.datasets.utils import download_and_extract_archive

from .utils import maybe_download

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


def get_labels_as_vector(class_names):
    classes = [CLASSNAME_TO_IDX[x] for x in class_names]
    label = torch.zeros(NUM_CLASSES, dtype=int)
    label[classes] = 1
    return label


class VOCMultiLabel(VOCDetection):
    def __init__(self, **kwargs):
        maybe_download(super().__init__, kwargs)

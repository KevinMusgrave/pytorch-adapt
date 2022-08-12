import os

import torch
from torchvision.datasets import VOCDetection
from torchvision.datasets.utils import download_and_extract_archive

from ..utils import common_functions as c_f
from .utils import check_length, maybe_download

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
def process_voc_style_dataset(
    cls, dataset_root, image_set, download, rename_fn=None, exclude_list=None
):
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

    exclude_list = c_f.default(exclude_list, [])
    file_names = [x for x in file_names if x not in exclude_list]
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

        check_length(
            self,
            {
                "2007": {"train": 2501, "trainval": 5011, "val": 2510, "test": 4952},
                "2008": {"train": 2111, "trainval": 4332, "val": 2221},
                "2009": {"train": 3473, "trainval": 7054, "val": 3581},
                "2010": {"train": 4998, "trainval": 10103, "val": 5105},
                "2011": {"train": 5717, "trainval": 11540, "val": 5823},
                "2012": {"train": 5717, "trainval": 11540, "val": 5823},
            }[self.year][self.image_set],
        )

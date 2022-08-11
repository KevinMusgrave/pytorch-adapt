import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from ..utils import common_functions as c_f
from .constants import IMAGENET_MEAN, IMAGENET_STD


def get_voc_transform(domain, train, is_training):
    bbox_params = A.BboxParams(
        format="pascal_voc", min_visibility=0.5, label_fields=["class_labels"]
    )
    transform = [A.SmallestMaxSize(224)]
    if is_training:
        transform += [
            A.RandomCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
        ]
    else:
        transform += [A.CenterCrop(height=224, width=224)]

    transform += [
        ToTensorV2(),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return A.Compose(transform, bbox_params=bbox_params)


def voc_transform_wrapper(transform, labels_to_vec):
    def fn(image, target):
        objects = target["annotation"]["object"]
        bboxes = [
            [int(y) for y in c_f.extract(x["bndbox"], ["xmin", "ymin", "xmax", "ymax"])]
            for x in objects
        ]
        class_labels = [x["name"] for x in objects]
        x = transform(image=np.array(image), bboxes=bboxes, class_labels=class_labels)
        image = x["image"]
        labels = labels_to_vec(x["class_labels"])
        return image, labels

    return fn

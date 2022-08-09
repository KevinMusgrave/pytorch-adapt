import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from ..utils import common_functions as c_f


def get_voc_transform():
    return A.Compose(
        [
            A.SmallestMaxSize(224),
            A.RandomCrop(width=224, height=224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.5, label_fields=["class_labels"]
        ),
    )


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

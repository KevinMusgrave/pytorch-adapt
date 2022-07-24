import torch
from torchvision.datasets import VOCDetection


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

    def __getitem__(self, idx):
        img, info = super().__getitem__(idx)
        classes = [
            self.CLASSNAME_TO_IDX[x["name"]] for x in info["annotation"]["object"]
        ]
        label = torch.zeros(self.NUM_CLASSES, dtype=int)
        label[classes] = 1
        return img, label

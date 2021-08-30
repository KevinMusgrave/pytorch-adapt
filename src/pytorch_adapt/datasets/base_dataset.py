import torch
from PIL import Image

from ..utils import common_functions as c_f


class BaseDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, label

    def __repr__(self):
        extra_repr = f"len={str(self.__len__())}"
        return c_f.nice_repr(self, extra_repr, {"transform": self.transform})

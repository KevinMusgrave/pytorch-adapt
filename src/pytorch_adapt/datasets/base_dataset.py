import os
import tarfile

import torch
from PIL import Image
from torchvision.datasets.utils import download_url

from ..utils import common_functions as c_f


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, domain):
        super().__init__()
        self.domain = domain

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        extra_repr = f"domain={self.domain}\nlen={str(self.__len__())}"
        return c_f.nice_repr(self, extra_repr, {"transform": self.transform})


class BaseDownloadableDataset(BaseDataset):
    def __init__(self, root, download=False, **kwargs):
        super().__init__(**kwargs)
        if download:
            try:
                self.set_paths_and_labels(root)
            except (FileNotFoundError, ValueError):
                self.download_dataset(root)
                self.set_paths_and_labels(root)
        else:
            self.set_paths_and_labels(root)

    def set_paths_and_labels(self, root):
        raise NotImplementedError

    def download_dataset(self, root):
        download_url(self.url, root, filename=self.filename, md5=self.md5)
        with tarfile.open(os.path.join(root, self.filename), "r:gz") as tar:
            tar.extractall(path=root, members=c_f.extract_progress(tar))

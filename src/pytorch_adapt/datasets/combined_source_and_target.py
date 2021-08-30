import numpy as np
import torch

from ..utils import common_functions as c_f


class CombinedSourceAndTargetDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return len(self.target_dataset)

    def __getitem__(self, idx):
        target_data = self.target_dataset[idx]
        src_data = self.source_dataset[self.get_random_src_idx(idx)]
        return {**src_data, **target_data}

    def get_random_src_idx(self, idx):
        src_len = len(self.source_dataset)
        random_offset = np.random.choice(src_len)
        return (idx + random_offset) % src_len

    def __repr__(self):
        return c_f.nice_repr(
            self,
            "",
            {
                "source_dataset": self.source_dataset,
                "target_dataset": self.target_dataset,
            },
        )

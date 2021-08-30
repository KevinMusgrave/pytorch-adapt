import torch

from ..utils import common_functions as c_f


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, domain):
        self.dataset = dataset
        self.domain = domain

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return c_f.nice_repr(
            self, c_f.extra_repr(self, ["domain"]), {"dataset": self.dataset}
        )

from torch.utils.data import Dataset

from ..utils import common_functions as c_f


class DomainDataset(Dataset):
    def __init__(self, dataset: Dataset, domain: int):
        self.dataset = dataset
        self.domain = domain

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return c_f.nice_repr(
            self, c_f.extra_repr(self, ["domain"]), {"dataset": self.dataset}
        )

import torch

from .domain_dataset import DomainDataset


class TargetDataset(DomainDataset):
    def __init__(self, dataset, domain=1):
        super().__init__(dataset, domain)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return {"target_imgs": img, "target_domain": self.domain, "sample_idx": idx}

import torch

from .domain_dataset import DomainDataset


class SourceDataset(DomainDataset):
    def __init__(self, dataset, domain=0):
        super().__init__(dataset, domain)

    def __getitem__(self, idx):
        img, src_labels = self.dataset[idx]
        return {
            "src_imgs": img,
            "src_domain": self.domain,
            "src_labels": src_labels,
            "src_sample_idx": idx,
        }

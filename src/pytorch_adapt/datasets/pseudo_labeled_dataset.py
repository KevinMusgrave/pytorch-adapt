import torch

from .domain_dataset import DomainDataset


class PseudoLabeledDataset(DomainDataset):
    def __init__(self, dataset, pseudo_labels, domain=0):
        super().__init__(dataset, domain)
        if len(self.dataset) != len(pseudo_labels):
            raise ValueError("len(dataset) must equal len(pseudo_labels)")
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return {
            "src_imgs": img,
            "src_domain": self.domain,
            "src_labels": self.pseudo_labels[idx],
            "src_sample_idx": idx,
        }

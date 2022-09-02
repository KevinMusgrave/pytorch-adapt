from typing import Any, Dict, List

from torch.utils.data import Dataset

from .domain_dataset import DomainDataset


class PseudoLabeledDataset(DomainDataset):
    """
    The wrapped dataset's ```__getitem__``` function
    should return a tuple of ```(data, label)```.
    The label will then be discarded,
    and the pseudo label will be returned instead.
    """

    def __init__(self, dataset: Dataset, pseudo_labels: List[int], domain: int = 0):
        """
        Arguments:
            dataset: The dataset to wrap
            pseudo_labels: The class labels that will be used
                instead of the labels contained in self.dataset
            domain: An integer representing the domain.
        """

        super().__init__(dataset, domain)
        if len(self.dataset) != len(pseudo_labels):
            raise ValueError("len(dataset) must equal len(pseudo_labels)")
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            A dictionary with keys

                - "src_imgs" (the data)

                - "src_domain" (the integer representing the domain)

                - "src_labels" (the pseudo label)

                - "src_sample_idx" (idx)
        """

        img, _ = self.dataset[idx]
        return {
            "src_imgs": img,
            "src_domain": self.domain,
            "src_labels": self.pseudo_labels[idx],
            "src_sample_idx": idx,
        }

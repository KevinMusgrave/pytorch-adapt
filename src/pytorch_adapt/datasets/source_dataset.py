from typing import Any, Dict

from torch.utils.data import Dataset

from .domain_dataset import DomainDataset


class SourceDataset(DomainDataset):
    """
    This wrapper returns a dictionary,
    but it expects the wrapped dataset to return a tuple of ```(data, label)```.
    """

    def __init__(self, dataset: Dataset, domain: int = 0):
        """
        Arguments:
            dataset: The dataset to wrap
            domain: An integer representing the domain.
        """
        super().__init__(dataset, domain)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            A dictionary with keys:

            - "src_imgs" (the data)

            - "src_domain" (the integer representing the domain)

            - "src_labels" (the class label)

            - "src_sample_idx" (idx)
        """

        img, src_labels = self.dataset[idx]
        return {
            "src_imgs": img,
            "src_domain": self.domain,
            "src_labels": src_labels,
            "src_sample_idx": idx,
        }

from typing import Any, Dict

from torch.utils.data import Dataset

from .domain_dataset import DomainDataset


class TargetDataset(DomainDataset):
    """
    Wrap your target dataset with this.

    If ```supervised = True```, the wrapped dataset's ```__getitem__```
    must return a tuple of ```(data, label)```.
    Otherwise it can return either ```(data, label)``` or ```data```.
    """

    def __init__(self, dataset: Dataset, domain: int = 1, supervised: bool = False):
        """
        Arguments:
            dataset: The dataset to wrap
            domain: An integer representing the domain.
            supervised: A boolean for if the target dataset should return labels.
        """
        super().__init__(dataset, domain)
        self.supervised = supervised

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            A dictionary with keys

            - "target_imgs" (the data)

            - "target_domain" (the integer representing the domain)

            - "target_sample_idx" (idx)

            If ```supervised = True``` it returns an extra key

            - "target_labels" (the class label)
        """

        has_labels = False
        img = self.dataset[idx]
        if isinstance(img, (list, tuple)):
            has_labels = True
            img, labels = img

        if self.supervised and not has_labels:
            raise ValueError(
                "if TargetDataset is instantiated with supervised=True, the wrapped dataset must include labels"
            )

        item = {
            "target_imgs": img,
            "target_domain": self.domain,
            "target_sample_idx": idx,
        }

        if self.supervised:
            item["target_labels"] = labels

        return item

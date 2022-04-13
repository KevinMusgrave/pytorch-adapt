from typing import Any, Dict

from torch.utils.data import Dataset

from .domain_dataset import DomainDataset


class TargetDataset(DomainDataset):
    """
    Wrap your target dataset with this. Your target dataset's
    ```__getitem__``` function should return a tuple of ```(data, label)```.
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

            If supervised = True it return an extra key

            - "target_labels (the class label)
        """
        
        img = self.dataset[idx]
        if isinstance(img, (list, tuple)):
            img, labels = img

        item = {
            "target_imgs": img,
            "target_domain": self.domain,
            "target_sample_idx": idx,
        }

        if self.supervised:
            item["target_labels"] = labels

        return item

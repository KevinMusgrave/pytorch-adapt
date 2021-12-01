from typing import Any, Dict

import numpy as np
import torch

from ..utils import common_functions as c_f
from .source_dataset import SourceDataset
from .target_dataset import TargetDataset


class CombinedSourceAndTargetDataset(torch.utils.data.Dataset):
    """
    Wraps a source dataset and a target dataset.
    """

    def __init__(self, source_dataset: SourceDataset, target_dataset: TargetDataset):
        """
        Arguments:
            source_dataset:
            target_dataset:
        """

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self) -> int:
        """
        Returns:
            The length of the target dataset.
        """
        return len(self.target_dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Arguments:
            idx: The index of the target dataset. The source index is picked randomly.
        Returns:
            A dictionary containing both source and target data.
            The source keys start with "src", and the target keys start with "target".
        """
        target_data = self.target_dataset[idx]
        src_data = self.source_dataset[self.get_random_src_idx()]
        return c_f.assert_dicts_are_disjoint(src_data, target_data)

    def get_random_src_idx(self):
        return np.random.choice(len(self.source_dataset))

    def __repr__(self):
        return c_f.nice_repr(
            self,
            "",
            {
                "source_dataset": self.source_dataset,
                "target_dataset": self.target_dataset,
            },
        )

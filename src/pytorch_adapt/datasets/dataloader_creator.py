from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from ..utils import common_functions as c_f


class DataloaderCreator:
    """
    This is a factory class for creating dataloaders.
    The ```__call__``` function takes in keyword arguments which are datasets,
    and outputs a dictionary of dataloaders (one dataloader for each input dataset).
    """

    def __init__(
        self,
        train_kwargs: Dict[str, Any] = None,
        val_kwargs: Dict[str, Any] = None,
        train_names: List[str] = None,
        val_names: List[str] = None,
        all_train: bool = False,
        all_val: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Arguments:
            train_kwargs: The keyword arguments that will be
                passed to every DataLoader constructor for train-time datasets.
                If ```None```, it defaults to:
                ```python
                {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "shuffle": True,
                    "drop_last": True,
                },
                ```
            val_kwargs: The keyword arguments that will be
                passed to every DataLoader constructor for validation-time datasets.
                If ```None```, it defaults to:
                ```python
                {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "shuffle": False,
                    "drop_last": False,
                }
                ```
            train_names: A list of the dataset names that are used during training.
                If ```None```, it defaults to ```["train"]```.
            val_names: A list of the dataset names that are used during validation.
                If ```None```, it defaults to ```["src_train", "target_train", "src_val", "target_val"]```.
            all_train: If True, then all input datasets are assumed to be for training,
                regardless of their names.
            all_val: If True, then all input datasets are assumed to be for validation,
                regardless of their names.
            batch_size: The ```batch_size``` used in the default ```train_kwargs```
                and in the default ```val_kwargs```.
            num_workers: The ```num_workers``` used in the default ```train_kwargs```
                and in the default ```val_kwargs```.
        """

        self.train_kwargs = c_f.default(
            train_kwargs,
            {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": True,
                "drop_last": True,
            },
        )
        self.val_kwargs = c_f.default(
            val_kwargs,
            {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": False,
                "drop_last": False,
            },
        )
        self.train_names = c_f.default(train_names, ["train"])
        self.val_names = c_f.default(
            val_names, ["src_train", "target_train", "src_val", "target_val"]
        )
        if not set(self.train_names).isdisjoint(self.val_names):
            raise ValueError(
                f"train_names {self.train_names} must be disjoint from val_names {self.val_names}"
            )
        if all_train and all_val:
            raise ValueError("all_train and all_val cannot both be True")
        self.all_train = all_train
        self.all_val = all_val

    def __call__(self, **kwargs) -> Dict[str, DataLoader]:
        """
        Arguments:
            **kwargs: keyword arguments mapping from dataset names to datasets.
        Returns:
            a dictionary mapping from dataset names to dataloaders.
        """

        output = {}
        for k, v in kwargs.items():
            if self.all_train:
                dataloader_kwargs = self.train_kwargs
            elif self.all_val:
                dataloader_kwargs = self.val_kwargs
            elif k in self.train_names:
                dataloader_kwargs = self.train_kwargs
            elif k in self.val_names:
                dataloader_kwargs = self.val_kwargs
            else:
                raise ValueError(
                    f"Dataset split name must be in {self.train_names} or {self.val_names}, or one of self.all_train or self.all_val must be true"
                )
            output[k] = torch.utils.data.DataLoader(v, **dataloader_kwargs)
        return output

import torch

from ..utils import common_functions as c_f


class DataloaderCreator:
    def __init__(
        self,
        train_kwargs=None,
        val_kwargs=None,
        train_names=None,
        val_names=None,
        all_train=False,
        all_val=False,
        batch_size=32,
        num_workers=0,
    ):
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

    def __call__(self, **kwargs):
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

import math
import unittest

import torch
from pytorch_metric_learning.utils.common_functions import EmbeddingDataset

from pytorch_adapt.datasets import DataloaderCreator


class TestDataloaderCreator(unittest.TestCase):
    def test_dataloader_creator(self):
        dataset_size = 1000
        emb_size = 128
        split_names = [
            "src_train",
            "src_val",
            "target_train",
            "target_val",
            "train",
        ]

        for train_names in [None, ["src_val", "target_train"]]:
            if train_names is None:
                val_names = None
            else:
                val_names = list(set(split_names) - set(train_names))
            for batch_size in [32, 64, 128]:
                datasets = {}
                bad_datasets = {}

                bad_split_names = split_names + ["test"]

                for name in split_names:
                    datasets[name] = EmbeddingDataset(
                        torch.randn(dataset_size, emb_size),
                        torch.randint(0, 10, size=(dataset_size,)),
                    )

                for name in bad_split_names:
                    bad_datasets[name] = EmbeddingDataset(
                        torch.randn(dataset_size, emb_size),
                        torch.randint(0, 10, size=(dataset_size,)),
                    )

                DC = DataloaderCreator(
                    train_names=train_names,
                    val_names=val_names,
                    batch_size=batch_size,
                )

                with self.assertRaises(ValueError):
                    dataloaders = DC(**bad_datasets)

                dataloaders = DC(**datasets)

                for name in split_names:
                    curr_dataloader = dataloaders[name]
                    if train_names is None:
                        is_random = name == "train"
                    else:
                        is_random = name in train_names
                    if not is_random:
                        self.assertTrue(
                            len(curr_dataloader) == math.ceil(dataset_size / batch_size)
                        )
                    for i, (embs, labels) in enumerate(curr_dataloader):
                        start = i * batch_size
                        end = i * batch_size + batch_size
                        correct_embs, correct_labels = datasets[name][start:end]
                        emb_is_equal = torch.equal(correct_embs, embs)
                        label_is_equal = torch.equal(correct_labels, labels)
                        self.assertTrue(is_random ^ emb_is_equal)
                        self.assertTrue(is_random ^ label_is_equal)

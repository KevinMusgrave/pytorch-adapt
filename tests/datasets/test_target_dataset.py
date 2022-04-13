import unittest

import numpy as np
import torch

from pytorch_adapt.datasets import TargetDataset


class TestTargetDataset(unittest.TestCase):
    def test_supervised_flag(self):
        target_dataset_size = 100
        tgt = torch.arange(target_dataset_size)
        
        tgt_unlabelled = [i for i in tgt] # just images
        tgt_labelled = [(i, i) for i in tgt] # images and labels
        
        """
            TargetDataset construction types

            +--------------------------+------------+-------------+----------------+
            |    TargetDatasetType     |  Dataset   | Supervised  |    Returns     |
            +--------------------------+------------+-------------+----------------+
            | Real world unsupervised  | unlabelled | false       | images         |
            | Academic unsupervised    | labelled   | false       | images, _      |
            | Real/Academic supervised | labelled   | true        | images, labels |
            | Error state              | unlabelled | true        | Error          |
            +--------------------------+------------+-------------+----------------+

        """
        sample_idx = torch.randint(len(tgt), size=(1,)).item()
        realworld_unsupervised = TargetDataset(tgt_unlabelled, supervised=False)
        item = realworld_unsupervised[sample_idx]
        # Convert tensors to compare using assert dict.
        item["target_imgs"] = item["target_imgs"].item()
        self.assertDictEqual(item, {
            "target_imgs": sample_idx,
            "target_domain": 1,
            "target_sample_idx": sample_idx
        })
        
        sample_idx = torch.randint(len(tgt), size=(1,)).item()
        academic_unsupervised = TargetDataset(tgt_labelled, supervised=False)
        item = academic_unsupervised[sample_idx]
        item["target_imgs"] = item["target_imgs"].item()
        self.assertDictEqual(item, {
            "target_imgs": sample_idx,
            "target_domain": 1,
            "target_sample_idx": sample_idx
        })
        
        sample_idx = torch.randint(len(tgt), size=(1,)).item()
        supervised = TargetDataset(tgt_labelled, supervised=True)
        item = supervised[sample_idx]
        item["target_imgs"] = item["target_imgs"].item()
        item["target_imgs"] = item["target_labels"].item()
        self.assertDictEqual(item, {
            "target_imgs": sample_idx,
            "target_domain": 1,
            "target_sample_idx": sample_idx,
            "target_labels": sample_idx
        })            
import unittest

import numpy as np
import torch

from pytorch_adapt.datasets import CombinedSourceAndTargetDataset
from pytorch_adapt.utils.common_functions import join_lists


class TestCombinedSourceAndTarget(unittest.TestCase):
    def test_combined(self):

        for target_dataset_size in [99, 199]:
            src_dataset_size = 117

            src = torch.arange(src_dataset_size)
            src = [{"src_imgs": i, "src_labels": i} for i in src]
            tgt = torch.arange(target_dataset_size)
            tgt = [{"target_imgs": i} for i in tgt]
            d = CombinedSourceAndTargetDataset(src, tgt)

            collected = []
            num_loops = 10000
            batch_size = 64
            total_len = num_loops * batch_size
            for x in range(num_loops):
                collected.append([])
                for i in range(batch_size):
                    batch = d[i]
                    collected[x].append(
                        (batch["src_imgs"].item(), batch["target_imgs"].item())
                    )

            all_src = []
            for c in collected:
                self.assertTrue([x[1] for x in c] == list(range(batch_size)))
                curr_src = [x[0] for x in c]
                # check for randomness
                self.assertTrue(curr_src not in all_src)
                all_src.append(curr_src)

            all_src = join_lists(all_src)
            self.assertTrue(len(all_src) == total_len)
            bincount = np.bincount(all_src)
            self.assertTrue(len(bincount) == src_dataset_size)
            ideal_bincount = total_len // src_dataset_size
            self.assertTrue(
                all(np.isclose(x, ideal_bincount, rtol=5e-2) for x in bincount)
            )

import unittest

import torch

from pytorch_adapt.datasets import CombinedSourceAndTargetDataset


class TestCombinedSourceAndTarget(unittest.TestCase):
    def test_combined(self):

        src = torch.arange(100)
        src = [{"src_imgs": i, "src_labels": i} for i in src]
        tgt = torch.arange(100)
        tgt = [{"target_imgs": i} for i in tgt]
        d = CombinedSourceAndTargetDataset(src, tgt)

        collected = []
        num_loops = 10
        inner_loop = 100
        for x in range(num_loops):
            collected.append([])
            for i in range(inner_loop):
                batch = d[i]
                collected[x].append(
                    (batch["src_imgs"].item(), batch["target_imgs"].item())
                )

        all_src = []
        for c in collected:
            self.assertTrue([x[1] for x in c] == list(range(inner_loop)))
            curr_src = [x[0] for x in c]
            self.assertTrue(curr_src not in all_src)
            all_src.append(curr_src)

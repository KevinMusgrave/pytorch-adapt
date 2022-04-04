import copy
import unittest

import torch
from sklearn.metrics import adjusted_mutual_info_score

from pytorch_adapt.validators import ClassClusterValidator

from .. import TEST_DEVICE


class TestClassClusterValidator(unittest.TestCase):
    def test_class_cluster_validator(self):
        def target_label_fn(x):
            return x["labels"]

        for pca_size in [None, 64]:
            for with_src in [False, True]:
                for src_for_pca in [False, True]:
                    if with_src and src_for_pca:
                        continue
                    for centroid_init in [None, "label_centers"]:
                        features, logits, labels = [], [], []
                        num_classes = 5
                        for i in range(num_classes):
                            features.append(
                                torch.randn(32, 128, device=TEST_DEVICE) + i * 100
                            )
                            logits.append(
                                torch.randn(32, num_classes, device=TEST_DEVICE)
                            )
                            labels.append(torch.ones(32, device=TEST_DEVICE) * i)

                        features = torch.cat(features, dim=0)
                        labels = torch.cat(labels, dim=0)
                        logits = torch.cat(logits, dim=0)
                        validator = ClassClusterValidator(
                            target_label_fn=target_label_fn,
                            pca_size=pca_size,
                            score_fn=adjusted_mutual_info_score,
                            with_src=with_src,
                            src_for_pca=src_for_pca,
                            centroid_init=centroid_init,
                            score_fn_type="labels",
                        )

                        args = {
                            "features": features,
                            "logits": logits,
                            "labels": labels,
                        }
                        kwargs = {"target_train": args}
                        if with_src or src_for_pca:
                            kwargs["src_train"] = copy.deepcopy(args)
                            kwargs["src_train"]["labels"] = torch.flip(
                                kwargs["src_train"]["labels"], dims=(0,)
                            )
                        score = validator(**kwargs)
                        if not with_src:
                            self.assertTrue(score == 1)
                        else:
                            self.assertTrue(score < 0.7)

                        for k in kwargs.keys():
                            kwargs[k]["labels"] = torch.randint(
                                0, num_classes, size=kwargs[k]["labels"].shape
                            )
                        score = validator(**kwargs)
                        self.assertTrue(score < 0.1)

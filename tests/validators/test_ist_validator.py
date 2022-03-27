import unittest

import torch

from pytorch_adapt.layers import ISTLoss
from pytorch_adapt.validators import ISTValidator

from .. import TEST_DEVICE


class TestISTValidator(unittest.TestCase):
    def test_ist_validator(self):
        torch.manual_seed(106)
        embedding_size = 32
        for with_ent in [True, False]:
            for with_div in [True, False]:
                if not (with_ent or with_div):
                    continue
                for batch_size in [33, 257, 999, 1000, 1001]:
                    for dataset_size in [10, 100, 1000, 2000]:
                        validator = ISTValidator(
                            batch_size=batch_size, with_ent=with_ent, with_div=with_div
                        )
                        src_features = torch.randn(dataset_size, embedding_size).to(
                            TEST_DEVICE
                        )
                        target_features = (
                            torch.randn(dataset_size, embedding_size).to(TEST_DEVICE)
                            + 0.1
                        )
                        src_domain = torch.zeros(len(src_features)).to(TEST_DEVICE)
                        target_domain = torch.ones(len(target_features)).to(TEST_DEVICE)

                        score = validator(
                            src_train={"features": src_features, "domain": src_domain},
                            target_train={
                                "features": target_features,
                                "domain": target_domain,
                            },
                        )

                        features = torch.cat([src_features, target_features], dim=0)
                        labels = torch.cat([src_domain, target_domain], dim=0)

                        loss = ISTLoss(with_ent=with_ent, with_div=with_div)(
                            features, labels
                        )
                        self.assertTrue(torch.isclose(loss, score, rtol=1e-2))

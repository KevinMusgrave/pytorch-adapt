import unittest

import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import LpDistance

from pytorch_adapt.layers import ISTLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestISTLoss(unittest.TestCase):
    def test_ist_loss(self):
        batch_size = 32
        embedding_size = 100
        for with_div in [True, False]:
            for distance in [None, LpDistance()]:
                loss_fn = ISTLoss(distance=distance, with_div=with_div)
                for dtype in TEST_DTYPES:
                    if dtype == torch.float16:
                        continue
                    x = torch.randn(
                        batch_size,
                        embedding_size,
                        device=TEST_DEVICE,
                        requires_grad=True,
                    ).type(dtype)
                    x.retain_grad()
                    y = torch.randint(0, 2, size=(batch_size,), device=TEST_DEVICE)
                    loss = loss_fn(x, y)
                    loss.backward()

                    ents, all_logits = [], []
                    for i in range(len(x)):
                        curr_x = x[i].unsqueeze(0)
                        mask = torch.ones(len(x), dtype=torch.bool)
                        mask[i] = False
                        other_xs = x[mask]

                        curr_x = F.normalize(curr_x, dim=1)
                        other_xs = F.normalize(other_xs, dim=1)
                        if distance is None:
                            dists = torch.mm(curr_x, other_xs.t())
                        else:
                            dists = -torch.sqrt(
                                torch.sum((curr_x - other_xs) ** 2, dim=1)
                            )
                        dists = dists.squeeze(0)
                        dists = F.softmax(dists, dim=0)
                        probs = dists * (y[mask])
                        target_prob = torch.sum(probs)
                        src_prob = 1 - target_prob
                        ent = -(
                            src_prob * torch.log(src_prob)
                            + target_prob * torch.log(target_prob)
                        )
                        ents.append(ent)

                        logits = torch.tensor([[src_prob, target_prob]])
                        all_logits.append(logits)

                    ent_loss = -(sum(ents) / len(ents))
                    if with_div:
                        all_logits = torch.cat(all_logits, dim=0)
                        mean_logits = torch.mean(all_logits, dim=0)
                        div = mean_logits[0] * torch.log(mean_logits[0]) + mean_logits[
                            1
                        ] * torch.log(mean_logits[1])
                        correct_loss = -div + ent_loss
                    else:
                        correct_loss = ent_loss

                    self.assertTrue(torch.isclose(loss, correct_loss, rtol=1e-3))

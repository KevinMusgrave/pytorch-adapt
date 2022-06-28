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
        for with_ent in [True, False]:
            for with_div in [True, False]:
                if not (with_ent or with_div):
                    continue
                for distance in [None, LpDistance()]:
                    loss_fn = ISTLoss(
                        distance=distance, with_ent=with_ent, with_div=with_div
                    )
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

                        ents, all_preds = [], []
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
                            curr_y = y[mask]
                            target_prob = torch.sum(dists[curr_y == 1])
                            src_prob = torch.sum(dists[curr_y == 0])
                            ent = -(
                                src_prob * torch.log(src_prob)
                                + target_prob * torch.log(target_prob)
                            )
                            ents.append(ent)

                            preds = torch.stack(
                                [src_prob.unsqueeze(0), target_prob.unsqueeze(0)], dim=1
                            )
                            all_preds.append(preds)

                        correct_loss = 0

                        if with_ent:
                            correct_loss += -(sum(ents) / len(ents))
                        if with_div:
                            all_preds = torch.cat(all_preds, dim=0)
                            mean_preds = torch.mean(all_preds, dim=0)
                            div = mean_preds[0] * torch.log(mean_preds[0]) + mean_preds[
                                1
                            ] * torch.log(mean_preds[1])
                            correct_loss += -div

                        self.assertAlmostEqual(
                            loss.item(), correct_loss.item(), places=6
                        )

                        loss.backward()
                        grad1 = x.grad.clone()
                        x.grad = None
                        correct_loss.backward()
                        grad2 = x.grad.clone()
                        self.assertTrue(torch.allclose(grad1, grad2, rtol=1e-6))

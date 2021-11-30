import copy
import unittest
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import DomainLossHook, EntropyReducer
from pytorch_adapt.layers import EntropyWeights, MinMaxNormalizer, SumNormalizer
from pytorch_adapt.utils import common_functions as c_f

from .utils import get_models_and_data, get_opts


class TestReducers(unittest.TestCase):
    def test_entropy_reducer(self):
        (
            G,
            C,
            D,
            src_imgs,
            src_labels,
            target_imgs,
            src_domain,
            target_domain,
        ) = get_models_and_data()

        opts = get_opts(G, D, C)

        originalG = copy.deepcopy(G)
        originalD = copy.deepcopy(D)
        originalC = copy.deepcopy(C)
        original_opts = get_opts(originalG, originalD, originalC)

        models = {"G": G, "C": C, "D": D}
        data = {
            "src_imgs": src_imgs,
            "target_imgs": target_imgs,
            "src_domain": src_domain,
            "target_domain": target_domain,
        }

        for normalizer in [SumNormalizer(), MinMaxNormalizer()]:
            for detach_weights in [True, False]:
                startC = copy.deepcopy(C)
                h1 = DomainLossHook(domains=["target"])
                entropy_weights_fn = EntropyWeights(normalizer=normalizer)
                h2 = EntropyReducer(
                    domains=["target"],
                    detach_weights=detach_weights,
                    entropy_weights_fn=entropy_weights_fn,
                )

                losses, outputs = h1({}, {**models, **data})
                losses, outputs = h2(losses, {**models, **outputs})

                [x.zero_grad() for x in opts]
                losses["target_domain_loss"].backward()
                [x.step() for x in opts]

                context = torch.no_grad() if detach_weights else nullcontext()

                target_features = originalG(target_imgs)
                target_logits = originalC(target_features)
                target_dlogits = originalD(target_features)

                target_domain_loss = F.binary_cross_entropy_with_logits(
                    target_dlogits, target_domain, reduction="none"
                )
                with context:
                    weights = -torch.sum(
                        F.softmax(target_logits, dim=1)
                        * F.log_softmax(target_logits, dim=1),
                        dim=1,
                    )
                    weights = 1 + torch.exp(-weights)
                    if isinstance(normalizer, SumNormalizer):
                        weights /= torch.sum(weights)
                    elif isinstance(normalizer, MinMaxNormalizer):
                        weights = (weights - torch.min(weights)) / (
                            torch.max(weights) - torch.min(weights)
                        )
                    else:
                        raise TypeError
                correct_loss = torch.mean(target_domain_loss * weights)

                self.assertTrue(
                    torch.isclose(correct_loss, losses["target_domain_loss"], rtol=1e-3)
                )

                [x.zero_grad() for x in original_opts]
                correct_loss.backward()
                [x.step() for x in original_opts]

                for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
                    self.assertTrue(
                        c_f.state_dicts_are_equal(
                            x.state_dict(), y.state_dict(), rtol=1e-2
                        )
                    )

                for x, y in [(startC, C), (startC, originalC)]:
                    is_equal = c_f.state_dicts_are_equal(x.state_dict(), y.state_dict())
                    if detach_weights:
                        self.assertTrue(is_equal)
                    else:
                        self.assertTrue(not is_equal)

import torch

from ...utils import common_functions as c_f
from .ignite import Ignite


class IgnitePredsAsFeatures(Ignite):
    def create_output_dict(self, features, logits):
        # features == logits == preds
        # don't include "logits" key because logits is actually preds
        if not torch.allclose(features, logits):
            raise ValueError(
                f"features and logits should be equal when using {c_f.cls_name(self)}"
            )
        return {
            "features": features,
            "preds": features,
        }

import torch

from ...utils import common_functions as c_f
from .ignite import Ignite


# This is for the case where
# G outputs preds
# C is Identity()
# In other words, the "features" are softmaxed logits
class IgnitePredsAsFeatures(Ignite):
    def create_output_dict(self, features, logits):
        if not torch.allclose(features, logits):
            raise ValueError(
                f"features and logits should be equal when using {c_f.cls_name(self)}"
            )
        return {
            "features": features,
            "preds": features,
        }

import torch

from ...utils import common_functions as c_f
from .ignite import Ignite


# This is for the case where
# G outputs logits
# C is Identity()
# D is torch.softmax() => rest of D
# In other words, the "features" are softmax
# But for the sake of convenience, G outputs logits instead of softmaxed logits
# So during validation, the features have to be softmaxed to get the actual features
class IgnitePredsAsFeatures(Ignite):
    def create_output_dict(self, features, logits):
        if not torch.allclose(features, logits):
            raise ValueError(
                f"features and logits should be equal when using {c_f.cls_name(self)}"
            )
        features = torch.softmax(logits, dim=1)
        return {
            "features": features,
            "logits": logits,
            "preds": features,
        }

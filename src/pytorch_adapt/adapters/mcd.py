from typing import Tuple

import torch

from ..hooks import MCDHook
from .base_adapter import BaseGCAdapter
from .utils import with_opt


class MCD(BaseGCAdapter):
    """
    Extends [BaseGCAdapter][pytorch_adapt.adapters.base_adapter.BaseGCAdapter]
    and wraps [MCDHook][pytorch_adapt.hooks.MCDHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|

    The C model must output a list of logits, where each list element
    corresponds with a separate classifier. Usually the number of
    classifiers is 2, so C should output ```[logits1, logits2]```.
    """

    hook_cls = MCDHook

    def inference_default(self, x, domain=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Features and logits, where ```logits = sum(C(features))```.
        """
        features = self.models["G"](x)
        logits_list = self.models["C"](features)
        logits = sum(logits_list)
        return features, logits

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=with_opt(["G"]), c_opts=with_opt(["C"]), **hook_kwargs
        )

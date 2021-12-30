from typing import Tuple

import torch

from ..hooks import SymNetsHook
from ..utils.common_functions import check_domain
from .base_adapter import BaseGCAdapter
from .utils import with_opt


class SymNets(BaseGCAdapter):
    """
    Extends [BaseGCAdapter][pytorch_adapt.adapters.base_adapter.BaseGCAdapter]
    and wraps [SymNetsHook][pytorch_adapt.hooks.SymNetsHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|

    The C model must output a list of logits: ```[logits1, logits2]```.
    """

    hook_cls = SymNetsHook

    def inference_default(self, x, domain) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: The input to the model
            domain: 0 for the source domain, 1 for the target domain.
        Returns:
            Features and logits, where ```logits = C(features)[domain]```.
        """
        domain = check_domain(self, domain)
        features = self.models["G"](x)
        logits = self.models["C"](features)[domain]
        return features, logits

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=with_opt(["G"]), c_opts=with_opt(["C"]), **hook_kwargs
        )

from typing import Tuple

import torch

from ..containers import KeyEnforcer
from ..hooks import AlignerPlusCHook, RTNHook
from ..utils.common_functions import check_domain
from .base_adapter import BaseGCAdapter
from .utils import with_opt


class Aligner(BaseGCAdapter):
    """
    Extends [BaseGCAdapter][pytorch_adapt.adapters.base_adapter.BaseGCAdapter]
    and wraps [AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|
    """

    hook_cls = AlignerPlusCHook

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts, **hook_kwargs)


class RTN(Aligner):
    """
    Extends [Aligner][pytorch_adapt.adapters.Aligner]
    and wraps [RTNHook][pytorch_adapt.hooks.RTNHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "residual_model"]```|
    |optimizers|```["G", "C", "residual_model"]```|
    |misc|```["feature_combiner"]```|
    """

    hook_cls = RTNHook

    def inference_default(self, x, domain=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: The input to the model
            domain: If 0, ```logits = residual_model(C(G(x)))```.
                Otherwise, ```logits = C(G(x))```.
        Returns:
            Features and logits
        """
        domain = check_domain(self, domain)
        features, logits = super().inference_default(x, domain)
        if domain == 0:
            return features, self.models["residual_model"](logits)
        return features, logits

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["models"].append("residual_model")
        ke.requirements["optimizers"].append("residual_model")
        ke.requirements["misc"] = ["feature_combiner"]
        return ke

from ..containers import KeyEnforcer
from ..hooks import AlignerPlusCHook, RTNHook
from ..inference import rtn_fn
from ..utils import common_functions as c_f
from .base_adapter import BaseGCAdapter
from .utils import with_opt


class Aligner(BaseGCAdapter):
    """
    Wraps [AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|
    """

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts, **hook_kwargs)

    @property
    def hook_cls(self):
        return AlignerPlusCHook


class RTN(Aligner):
    """
    Wraps [RTNHook][pytorch_adapt.hooks.RTNHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "residual_model"]```|
    |optimizers|```["G", "C", "residual_model"]```|
    |misc|```["feature_combiner"]```|
    """

    def __init__(self, *args, inference_fn=None, **kwargs):
        """
        Arguments:
            inference_fn: Default is [rtn_fn][pytorch_adapt.inference.rtn_fn]
        """
        inference_fn = c_f.default(inference_fn, rtn_fn)
        super().__init__(*args, inference_fn=inference_fn, **kwargs)

    @property
    def hook_cls(self):
        return RTNHook

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["models"].append("residual_model")
        ke.requirements["optimizers"].append("residual_model")
        ke.requirements["misc"] = ["feature_combiner"]
        return ke

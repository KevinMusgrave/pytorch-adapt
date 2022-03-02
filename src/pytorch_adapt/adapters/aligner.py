from ..containers import KeyEnforcer
from ..hooks import AlignerPlusCHook, RTNHook
from ..inference import rtn_fn
from ..utils import common_functions as c_f
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

    def __init__(self, *args, inference_fn=None, **kwargs):
        inference_fn = c_f.default(inference_fn, rtn_fn)
        super().__init__(*args, inference_fn=inference_fn, **kwargs)

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["models"].append("residual_model")
        ke.requirements["optimizers"].append("residual_model")
        ke.requirements["misc"] = ["feature_combiner"]
        return ke

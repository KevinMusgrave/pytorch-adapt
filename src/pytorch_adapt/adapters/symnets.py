from ..hooks import SymNetsHook
from ..inference import symnets_fn
from ..utils import common_functions as c_f
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

    def __init__(self, *args, inference_fn=None, **kwargs):
        inference_fn = c_f.default(inference_fn, symnets_fn)
        super().__init__(*args, inference_fn=inference_fn, **kwargs)

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=with_opt(["G"]), c_opts=with_opt(["C"]), **hook_kwargs
        )

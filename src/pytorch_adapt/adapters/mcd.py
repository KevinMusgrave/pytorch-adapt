from ..hooks import MCDHook
from ..inference import mcd_fn
from ..utils import common_functions as c_f
from .base_adapter import BaseGCAdapter
from .utils import with_opt


class MCD(BaseGCAdapter):
    """
    Wraps [MCDHook][pytorch_adapt.hooks.MCDHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|

    The C model must output a list of logits, where each list element
    corresponds with a separate classifier. Usually the number of
    classifiers is 2, so C should output ```[logits1, logits2]```.
    """

    def __init__(self, *args, inference_fn=None, **kwargs):
        inference_fn = c_f.default(inference_fn, mcd_fn)
        super().__init__(*args, inference_fn=inference_fn, **kwargs)

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=with_opt(["G"]), c_opts=with_opt(["C"]), **hook_kwargs
        )

    @property
    def hook_cls(self):
        return MCDHook

from ..hooks import MCDHook
from .base_adapter import BaseGCAdapter
from .utils import with_opt


class MCD(BaseGCAdapter):
    """
    Wraps [MCDHook][pytorch_adapt.hooks.mcd].
    """

    hook_cls = MCDHook

    def inference_default(self, x, domain=None):
        features = self.models["G"](x)
        logits_list = self.models["C"](features)
        logits = sum(logits_list)
        return features, logits

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=with_opt(["G"]), c_opts=with_opt(["C"]), **hook_kwargs
        )

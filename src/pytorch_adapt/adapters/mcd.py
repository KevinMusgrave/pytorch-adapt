from ..containers import KeyEnforcer
from ..hooks import MCDHook
from .base_adapter import BaseGCAdapter


class MCD(BaseGCAdapter):
    hook_cls = MCDHook

    def inference_default(self, x, domain=None):
        features = self.models["G"](x)
        logits_list = self.models["C"](features)
        logits = sum(logits_list)
        return features, logits

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=[self.optimizers["G"]], c_opts=[self.optimizers["C"]], **hook_kwargs
        )

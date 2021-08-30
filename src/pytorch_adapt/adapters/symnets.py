from ..containers import KeyEnforcer
from ..hooks import SymNetsHook
from ..utils.common_functions import check_domain
from .base_adapter import BaseGCAdapter


class SymNets(BaseGCAdapter):
    hook_cls = SymNetsHook

    def inference_default(self, x, domain):
        domain = check_domain(self, domain)
        features = self.models["G"](x)
        logits = self.models["C"](features)[domain]
        return features, logits

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            g_opts=[self.optimizers["G"]], c_opts=[self.optimizers["C"]], **hook_kwargs
        )

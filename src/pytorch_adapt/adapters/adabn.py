from ..containers import KeyEnforcer, MultipleContainers
from ..hooks import AdaBNHook
from ..utils.common_functions import check_domain
from .base_adapter import BaseAdapter


class AdaBN(BaseAdapter):
    hook_cls = AdaBNHook

    def inference_default(self, x, domain):
        domain = check_domain(self, domain, keep_len=True)
        features = self.models["G"](x, domain)
        logits = self.models["C"](features, domain)
        return features, logits

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(**hook_kwargs)

    def get_key_enforcer(self):
        return KeyEnforcer(models=["G", "C"], optimizers=[])

    def get_default_containers(self):
        return MultipleContainers()

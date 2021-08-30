from ..containers import KeyEnforcer
from ..hooks import DANNHook, GVBHook
from ..layers import ModelWithBridge
from .base_adapter import BaseGCDAdapter


class DANN(BaseGCDAdapter):
    hook_cls = DANNHook

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(opts=list(self.optimizers.values()), **hook_kwargs)


class GVB(DANN):
    hook_cls = GVBHook

    def init_containers_and_check_keys(self):
        models = self.containers["models"]
        for k in ["D", "C"]:
            if not isinstance(models[k], ModelWithBridge):
                models[k] = ModelWithBridge(models[k])
        super().init_containers_and_check_keys()

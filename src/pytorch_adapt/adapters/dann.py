from ..containers import KeyEnforcer
from ..hooks import CDANNEHook, DANNEHook, DANNHook, GVBEHook, GVBHook
from ..layers import ModelWithBridge
from .base_adapter import BaseGCDAdapter
from .gan import CDAN


class DANN(BaseGCDAdapter):
    """
    Wraps [DANNHook][pytorch_adapt.hooks.dann].
    """

    hook_cls = DANNHook

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(opts=list(self.optimizers.values()), **hook_kwargs)


class DANNE(DANN):
    hook_cls = DANNEHook


class CDANNE(DANN, CDAN):
    hook_cls = CDANNEHook


class GVB(DANN):
    """
    Wraps [GVBHook][pytorch_adapt.hooks.gvb].
    """

    hook_cls = GVBHook

    def init_containers_and_check_keys(self):
        models = self.containers["models"]
        for k in ["D", "C"]:
            if not isinstance(models[k], ModelWithBridge):
                models[k] = ModelWithBridge(models[k])
        super().init_containers_and_check_keys()


class GVBE(GVB):
    hook_cls = GVBEHook

from ..hooks import CDANNEHook, DANNEHook, DANNHook, GVBEHook, GVBHook
from ..layers import ModelWithBridge
from .base_adapter import BaseGCDAdapter
from .gan import CDAN
from .utils import with_opt


class DANN(BaseGCDAdapter):
    """
    Wraps [DANNHook][pytorch_adapt.hooks.dann].
    """

    hook_cls = DANNHook

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts=opts, **hook_kwargs)


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

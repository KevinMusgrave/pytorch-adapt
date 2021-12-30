from ..hooks import CDANNEHook, DANNEHook, DANNHook, GVBEHook, GVBHook
from ..layers import ModelWithBridge
from .base_adapter import BaseGCDAdapter
from .gan import CDAN
from .utils import with_opt


class DANN(BaseGCDAdapter):
    """
    Extends [BaseGCDAdapter][pytorch_adapt.adapters.base_adapter.BaseGCDAdapter]
    and wraps [DANNHook][pytorch_adapt.hooks.DANNHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
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
    Extends [DANN][pytorch_adapt.adapters.DANN]
    and wraps [GVBHook][pytorch_adapt.hooks.GVBHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|

    Models D and C must be of type [```ModelWithBridge```][pytorch_adapt.layers.ModelWithBridge].
    If not, they will be converted into instances of ```ModelWithBridge```,
    with each bridge being a re-initialized copy of each model.
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

from ..hooks import CDANNEHook, DANNEHook, DANNHook, GVBEHook, GVBHook
from ..layers import ModelWithBridge
from .base_adapter import BaseGCDAdapter
from .gan import CDAN
from .utils import with_opt


class DANN(BaseGCDAdapter):
    """
    Wraps [DANNHook][pytorch_adapt.hooks.DANNHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    """

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts=opts, **hook_kwargs)

    @property
    def hook_cls(self):
        return DANNHook


class DANNE(DANN):
    @property
    def hook_cls(self):
        return DANNEHook


class CDANNE(DANN, CDAN):
    @property
    def hook_cls(self):
        return CDANNEHook


class GVB(DANN):
    """
    Wraps [GVBHook][pytorch_adapt.hooks.GVBHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|

    Models D and C must be of type [```ModelWithBridge```][pytorch_adapt.layers.ModelWithBridge].
    If not, they will be converted into instances of ```ModelWithBridge```,
    with each bridge being a re-initialized copy of each model.
    """

    @property
    def hook_cls(self):
        return GVBHook

    def init_containers_and_check_keys(self, containers):
        models = containers["models"]
        for k in ["D", "C"]:
            if not isinstance(models[k], ModelWithBridge):
                models[k] = ModelWithBridge(models[k])
        super().init_containers_and_check_keys(containers)


class GVBE(GVB):
    @property
    def hook_cls(self):
        return GVBEHook

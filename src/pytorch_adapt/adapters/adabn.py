from ..containers import KeyEnforcer, MultipleContainers
from ..hooks import AdaBNHook
from ..inference import adabn_fn
from ..utils import common_functions as c_f
from .base_adapter import BaseAdapter


class AdaBN(BaseAdapter):
    """
    Wraps [AdaBNHook][pytorch_adapt.hooks.AdaBNHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    """

    def __init__(self, *args, inference_fn=None, **kwargs):
        """
        Arguments:
            inference_fn: Default is [adabn_fn][pytorch_adapt.inference.adabn_fn]
        """
        inference_fn = c_f.default(inference_fn, adabn_fn)
        super().__init__(*args, inference_fn=inference_fn, **kwargs)

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(**hook_kwargs)

    @property
    def hook_cls(self):
        return AdaBNHook

    def get_key_enforcer(self) -> KeyEnforcer:
        return KeyEnforcer(models=["G", "C"], optimizers=[])

    def get_default_containers(self) -> MultipleContainers:
        return MultipleContainers()

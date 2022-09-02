import copy

from ..containers import KeyEnforcer, MultipleContainers, Optimizers
from ..hooks import ADDAHook
from ..inference import adda_fn
from ..utils import common_functions as c_f
from .base_adapter import BaseAdapter
from .utils import default_optimizer_tuple, with_opt


class ADDA(BaseAdapter):
    """
    Wraps [ADDAHook][pytorch_adapt.hooks.ADDAHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["D", "T"]```|

    The target model ("T") is created during initialization by deep-copying the G model.
    """

    def __init__(self, *args, inference_fn=None, **kwargs):
        """
        Arguments:
            inference_fn: Default is [adda_fn][pytorch_adapt.inference.adda_fn]
        """
        inference_fn = c_f.default(inference_fn, adda_fn)
        super().__init__(*args, inference_fn=inference_fn, **kwargs)

    def get_default_containers(self) -> MultipleContainers:
        """
        Returns:
            The default set of containers. This will use the
                [default optimizer][pytorch_adapt.adapters.utils.default_optimizer_tuple]
                for the T and D models.
        """
        optimizers = Optimizers(default_optimizer_tuple(), keys=["T", "D"])
        return MultipleContainers(optimizers=optimizers)

    def get_key_enforcer(self) -> KeyEnforcer:
        return KeyEnforcer(
            models=["G", "C", "D", "T"],
            optimizers=["D", "T"],
        )

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            d_opts=with_opt(["D"]), g_opts=with_opt(["T"]), **hook_kwargs
        )

    def init_containers_and_check_keys(self, containers):
        containers["models"]["T"] = copy.deepcopy(containers["models"]["G"])
        super().init_containers_and_check_keys(containers)

    @property
    def hook_cls(self):
        return ADDAHook

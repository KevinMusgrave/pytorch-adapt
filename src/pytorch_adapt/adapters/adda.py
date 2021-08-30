import copy

import torch

from ..containers import KeyEnforcer, MultipleContainers, Optimizers
from ..hooks import ADDAHook
from ..utils.common_functions import check_domain
from .base_adapter import BaseAdapter
from .utils import default_optimizer_tuple


class ADDA(BaseAdapter):
    hook_cls = ADDAHook

    def inference_default(self, x, domain):
        domain = check_domain(self, domain)
        fe = "G" if domain == 0 else "T"
        features = self.models[fe](x)
        logits = self.models["C"](features)
        return features, logits

    def get_default_containers(self):
        optimizers = Optimizers(default_optimizer_tuple(), keys=["T", "D"])
        return MultipleContainers(optimizers=optimizers)

    def get_key_enforcer(self):
        return KeyEnforcer(
            models=["G", "C", "D", "T"],
            optimizers=["D", "T"],
        )

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(
            d_opts=[self.optimizers["D"]], g_opts=[self.optimizers["T"]], **hook_kwargs
        )

    def init_containers_and_check_keys(self):
        self.containers["models"]["T"] = copy.deepcopy(self.containers["models"]["G"])
        super().init_containers_and_check_keys()

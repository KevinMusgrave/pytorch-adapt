import torch

from ..containers import KeyEnforcer, MultipleContainers, Optimizers
from ..hooks import ClassifierHook, FinetunerHook
from .base_adapter import BaseGCAdapter
from .utils import default_optimizer_tuple


class Classifier(BaseGCAdapter):
    hook_cls = ClassifierHook

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(opts=list(self.optimizers.values()), **hook_kwargs)


class Finetuner(Classifier):
    hook_cls = FinetunerHook

    def get_default_containers(self):
        optimizers = Optimizers(default_optimizer_tuple(), keys=["C"])
        return MultipleContainers(optimizers=optimizers)

    def get_key_enforcer(self):
        ke = super().get_key_enforcer()
        ke.requirements["optimizers"].remove("G")
        return ke

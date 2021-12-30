from ..containers import KeyEnforcer, MultipleContainers, Optimizers
from ..hooks import ClassifierHook, FinetunerHook
from .base_adapter import BaseGCAdapter
from .utils import default_optimizer_tuple, with_opt


class Classifier(BaseGCAdapter):
    """
    Extends [BaseGCAdapter][pytorch_adapt.adapters.base_adapter.BaseGCAdapter]
    and wraps [ClassifierHook][pytorch_adapt.hooks.ClassifierHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|
    """

    hook_cls = ClassifierHook

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts=opts, **hook_kwargs)


class Finetuner(Classifier):
    """
    Extends [Classifier][pytorch_adapt.adapters.Classifier]
    and wraps [FinetunerHook][pytorch_adapt.hooks.FinetunerHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["C"]```|
    """

    hook_cls = FinetunerHook

    def get_default_containers(self) -> MultipleContainers:
        optimizers = Optimizers(default_optimizer_tuple(), keys=["C"])
        return MultipleContainers(optimizers=optimizers)

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["optimizers"].remove("G")
        return ke

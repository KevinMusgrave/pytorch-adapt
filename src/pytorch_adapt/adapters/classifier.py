from ..containers import KeyEnforcer, MultipleContainers, Optimizers
from ..hooks import (
    ClassifierHook,
    FinetunerHook,
    MultiLabelClassifierHook,
    MultiLabelFinetunerHook,
)
from .base_adapter import BaseGCAdapter
from .utils import default_optimizer_tuple, with_opt


class Classifier(BaseGCAdapter):
    """
    Wraps [ClassifierHook][pytorch_adapt.hooks.ClassifierHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|
    """

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts=opts, **hook_kwargs)

    @property
    def hook_cls(self):
        return ClassifierHook


class Finetuner(Classifier):
    """
    Wraps [FinetunerHook][pytorch_adapt.hooks.FinetunerHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["C"]```|
    """

    @property
    def hook_cls(self):
        return FinetunerHook

    def get_default_containers(self) -> MultipleContainers:
        optimizers = Optimizers(default_optimizer_tuple(), keys=["C"])
        return MultipleContainers(optimizers=optimizers)

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["optimizers"].remove("G")
        return ke


class MultiLabelClassifier(Classifier):
    @property
    def hook_cls(self):
        return MultiLabelClassifierHook


class MultiLabelFinetuner(Finetuner):
    @property
    def hook_cls(self):
        return MultiLabelFinetunerHook

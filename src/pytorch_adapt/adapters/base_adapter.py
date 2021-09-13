from abc import ABC, abstractmethod
from enum import Enum

import torch

from ..containers import KeyEnforcer, MultipleContainers, Optimizers
from ..utils import common_functions as c_f
from .utils import default_optimizer_tuple


class BaseAdapter(ABC):
    """
    Parent class of all adapters.
    """

    def __init__(
        self,
        models=None,
        optimizers=None,
        lr_schedulers=None,
        misc=None,
        default_containers=None,
        key_enforcer=None,
        inference=None,
        before_training_starts=None,
        hook_kwargs=None,
    ):
        """
        Arguments:
            models: A [```Models```][pytorch_adapt.containers.models] container.
                The models will be passed to the wrapped hook at each
                training iteration.
            optimizers: An [```Optimizers```][pytorch_adapt.containers.optimizers] container.
                The optimizers will be passed into the wrapped hook during
                initialization. The hook uses the optimizers at each training iteration.
            lr_schedulers: An [```LRSchedulers```][pytorch_adapt.containers.lr_schedulers] container.
                The lr schedulers are called automatically by the
                [```framework```](../frameworks/index.md) that wrap this adapter.
            misc: A [```Misc```][pytorch_adapt.containers.misc] container for any
                miscellaneous objects. These are passed into the wrapped hook
                at each training iteration.
            hook_kwargs: A dictionary of key word arguments that will be
                passed into the wrapped hook during initialization.
        """
        self.containers = c_f.default(
            default_containers, self.get_default_containers, {}
        )
        self.key_enforcer = c_f.default(key_enforcer, self.get_key_enforcer, {})
        self.before_training_starts = c_f.class_default(
            self, before_training_starts, self.before_training_starts_default
        )

        self.containers.merge(
            models=models,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            misc=misc,
        )

        hook_kwargs = c_f.default(hook_kwargs, {})
        self.init_containers_and_check_keys()
        self.init_hook(hook_kwargs)
        self.inference = c_f.class_default(self, inference, self.inference_default)

    def training_step(self, batch, device, framework=None):
        batch = c_f.batch_to_device(batch, device)
        c_f.assert_dicts_are_disjoint(self.models, self.misc, batch)
        losses, _ = self.hook({}, {**self.models, **self.misc, **batch})
        return losses

    def inference_default(self, x, domain=None):
        features = self.models["G"](x)
        logits = self.models["C"](features)
        return features, logits

    def get_default_containers(self):
        optimizers = Optimizers(default_optimizer_tuple())
        return MultipleContainers(optimizers=optimizers)

    @abstractmethod
    def get_key_enforcer(self):
        pass

    @abstractmethod
    def init_hook(self):
        pass

    def init_containers_and_check_keys(self):
        self.containers.create()
        self.key_enforcer.check(self.containers)
        for k, v in self.containers.items():
            setattr(self, k, v)

    def before_training_starts_default(self, framework):
        c_f.LOGGER.info(f"models\n{self.models}")
        c_f.LOGGER.info(f"optimizers\n{self.optimizers}")
        c_f.LOGGER.info(f"lr_schedulers\n{self.lr_schedulers}")
        c_f.LOGGER.info(f"misc\n{self.misc}")
        c_f.LOGGER.info(f"hook\n{self.hook}")


class BaseGCDAdapter(BaseAdapter):
    def get_key_enforcer(self):
        return KeyEnforcer(
            models=["G", "C", "D"],
            optimizers=["G", "C", "D"],
        )


class BaseGCAdapter(BaseAdapter):
    def get_key_enforcer(self):
        return KeyEnforcer(
            models=["G", "C"],
            optimizers=["G", "C"],
        )

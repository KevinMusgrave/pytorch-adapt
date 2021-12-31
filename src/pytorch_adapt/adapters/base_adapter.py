from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from ..containers import (
    KeyEnforcer,
    LRSchedulers,
    Misc,
    Models,
    MultipleContainers,
    Optimizers,
)
from ..utils import common_functions as c_f
from .utils import default_optimizer_tuple, with_opt


class BaseAdapter(ABC):
    """
    Parent class of all adapters.
    """

    def __init__(
        self,
        models: Models = None,
        optimizers: Optimizers = None,
        lr_schedulers: LRSchedulers = None,
        misc: Misc = None,
        default_containers: MultipleContainers = None,
        key_enforcer: KeyEnforcer = None,
        inference=None,
        before_training_starts=None,
        hook_kwargs: Dict[str, Any] = None,
    ):
        """
        Arguments:
            models: A [```Models```][pytorch_adapt.containers.Models] container.
                The models will be passed to the wrapped hook at each
                training iteration.
            optimizers: An [```Optimizers```][pytorch_adapt.containers.Optimizers] container.
                The optimizers will be passed into the wrapped hook during
                initialization. The hook uses the optimizers at each training iteration.
            lr_schedulers: An [```LRSchedulers```][pytorch_adapt.containers.LRSchedulers] container.
                The lr schedulers are called automatically by the
                [```framework```](../frameworks/index.md) that wrap this adapter.
            misc: A [```Misc```][pytorch_adapt.containers.misc] container for models
                that don't require optimizers and other miscellaneous objects.
                These are passed into the wrapped hook at each training iteration.
            default_containers: The default set of containers to use, wrapped in a
                [```MultipleContainers```][pytorch_adapt.containers.MultipleContainers] object.
                If ```None``` then the default containers are defined in
                [```self.get_default_containers```][pytorch_adapt.adapters.BaseAdapter.get_default_containers]
            key_enforcer: A [```KeyEnforcer```][pytorch_adapt.containers.KeyEnforcer] object.
                If ```None```, then [```self.get_key_enforcer```][pytorch_adapt.adapters.BaseAdapter.get_key_enforcer]
                is used.
            inference: A function that takes in this adapter and returns another function
                that will be used for inference (i.e. during testing).
            before_training_starts: A function that takes in this adapter and returns another
                function that is optionally called by a framework wrapper before training starts.
            hook_kwargs: A dictionary of keyword arguments that will be
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

    def training_step(
        self, batch: Dict[str, Any], **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Calls the wrapped hook at each iteration during training.
        Arguments:
            batch: A dictionary containing training data.
        Returns:
            A two-level dictionary:

            - the outer level is associated with a particular optimization step
                (relevant for GAN architectures),

            - the inner level contains the loss components.
        """
        combined = c_f.assert_dicts_are_disjoint(
            self.models, self.misc, with_opt(self.optimizers), batch, kwargs
        )
        losses, _ = self.hook({}, combined)
        return losses

    def inference_default(
        self, x: torch.Tensor, domain: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: The input to the model
            domain: An optional integer indicating the domain.

        Returns:
            Features and logits
        """
        features = self.models["G"](x)
        logits = self.models["C"](features)
        return features, logits

    def get_default_containers(self) -> MultipleContainers:
        """
        Returns:
            The default set of containers. This will create
            an Adam optimizer with lr 0.0001 for
            each model that is passed into ```__init__```.
        """
        optimizers = Optimizers(default_optimizer_tuple())
        return MultipleContainers(optimizers=optimizers)

    @abstractmethod
    def get_key_enforcer(self) -> KeyEnforcer:
        """
        Returns:
            The default KeyEnforcer.
        """
        pass

    @abstractmethod
    def init_hook(self):
        """
        ```self.hook``` is initialized here.
        """
        pass

    def init_containers_and_check_keys(self):
        """
        Called in ```__init__```.
        """
        self.containers.create()
        self.key_enforcer.check(self.containers)
        for k, v in self.containers.items():
            setattr(self, k, v)

    def before_training_starts_default(self, framework):
        c_f.LOGGER.debug(f"models\n{self.models}")
        c_f.LOGGER.debug(f"optimizers\n{self.optimizers}")
        c_f.LOGGER.debug(f"lr_schedulers\n{self.lr_schedulers}")
        c_f.LOGGER.debug(f"misc\n{self.misc}")
        c_f.LOGGER.debug(f"hook\n{self.hook}")


class BaseGCDAdapter(BaseAdapter):
    """
    Base class for adapters that use a Generator, Classifier, and Discriminator.
    """

    def get_key_enforcer(self) -> KeyEnforcer:
        return KeyEnforcer(
            models=["G", "C", "D"],
            optimizers=["G", "C", "D"],
        )


class BaseGCAdapter(BaseAdapter):
    """
    Base class for adapters that use a Generator and Classifier.
    """

    def get_key_enforcer(self) -> KeyEnforcer:
        return KeyEnforcer(
            models=["G", "C"],
            optimizers=["G", "C"],
        )

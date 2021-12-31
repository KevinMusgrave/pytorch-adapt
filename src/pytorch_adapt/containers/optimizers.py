import copy
from typing import List

from ..layers import DoNothingOptimizer
from ..utils import common_functions as c_f
from .base_container import BaseContainer


class Optimizers(BaseContainer):
    """
    A container for model optimizers.
    """

    def __init__(self, *args, multipliers=None, **kwargs):
        """
        Arguments:
            *args: [```BaseContainer```][pytorch_adapt.containers.base_container.BaseContainer] arguments.
            multipliers: A dictionary mapping from
                optimizer name to lr multiplier. Each
                optimizer will have ```lr = lr * multiplier```
                upon initialization. If ```None```,
                then multiplier is 1.
            **kwargs:  [```BaseContainer```][pytorch_adapt.containers.base_container.BaseContainer]
                keyword arguments.
        """
        self.multipliers = c_f.default(multipliers, {})
        super().__init__(*args, **kwargs)

    def _create_with(self, other):
        c_f.assert_keys_are_present_cls(self, "multipliers", self)
        for k, v in self.items():
            if c_f.is_optimizer(v):
                continue
            class_ref, kwargs = v
            model = other[k]
            if c_f.has_no_parameters(model):
                self[k] = DoNothingOptimizer()
            else:
                kwargs = copy.deepcopy(kwargs)
                kwargs["lr"] *= self.multipliers.get(k, 1)
                self[k] = class_ref(model.parameters(), **kwargs)

    def step(self):
        """
        Calls ```.step()``` on all optimizers.
        """
        for v in self.values():
            v.step()

    def zero_grad(self):
        """
        Calls ```.zero_grad()``` on all optimizers.
        """
        for v in self.values():
            v.zero_grad()

    def merge(self, other):
        super().merge(other)
        self.multipliers.update(other.multipliers)

    def zero_back_step(self, loss, keys: List[str] = None):
        """
        Zeros gradients, computes gradients, and updates model weights.
        Arguments:
            loss: The loss on which ```.backward()``` is called.
            keys: The subset of optimizers on which to call
                ```.zero_grad()``` and ```.step()```.
                If ```None```, then all optimizers are used.
        """
        keys = c_f.default(keys, self.keys())
        optimizers = [self[k] for k in keys]
        c_f.zero_back_step(loss, optimizers)

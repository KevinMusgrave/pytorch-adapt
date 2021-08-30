import copy

from ..utils import common_functions as c_f
from .base_container import BaseContainer


class Optimizers(BaseContainer):
    def __init__(self, *args, multipliers=None, **kwargs):
        self.multipliers = c_f.default(multipliers, {})
        super().__init__(*args, **kwargs)

    def _create_with(self, other):
        to_be_deleted = []
        c_f.assert_keys_are_present_cls(self, "multipliers", self)
        for k, v in self.items():
            if c_f.is_optimizer(v):
                continue
            class_ref, kwargs = v
            model = other[k]
            if c_f.has_no_parameters(model):
                to_be_deleted.append(k)
            else:
                kwargs = copy.deepcopy(kwargs)
                kwargs["lr"] *= self.multipliers.get(k, 1)
                self[k] = class_ref(model.parameters(), **kwargs)

        for k in to_be_deleted:
            del self[k]

    def step(self):
        for v in self.values():
            v.step()

    def zero_grad(self):
        for v in self.values():
            v.zero_grad()

    def merge(self, other):
        super().merge(other)
        self.multipliers.update(other.multipliers)

    def zero_back_step(self, loss, keys=None):
        keys = c_f.default(keys, self.keys())
        optimizers = [self[k] for k in keys]
        c_f.zero_back_step(loss, optimizers)

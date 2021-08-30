from ..utils import common_functions as c_f
from .base_container import BaseContainer


class LRSchedulers(BaseContainer):
    def __init__(self, store, scheduler_types=None, **kwargs):
        self.scheduler_types = scheduler_types
        super().__init__(store, **kwargs)

    def _create_with(self, other):
        to_be_deleted = []
        for k, v in self.items():
            try:
                class_ref, kwargs = v
            except TypeError:
                continue
            optimizer = other[k]
            if not c_f.is_optimizer(optimizer):
                to_be_deleted.append(k)
            else:
                self[k] = class_ref(optimizer, **kwargs)

        for k in to_be_deleted:
            del self[k]

    def step(self, scheduler_type):
        for v in self.filter_by_scheduler_type(scheduler_type):
            v.step()

    def filter_by_scheduler_type(self, x):
        if self.scheduler_types is not None:
            return [v for k, v in self.items() if k in self.scheduler_types[x]]
        elif x == "per_step":
            return self.values()
        return []

    def merge(self, other):
        super().merge(other)
        if other.scheduler_types is not None:
            if self.scheduler_types is not None:
                for k, v in other.scheduler_types.items():
                    curr_list = self.scheduler_types[k]
                    curr_list.extend(v)
                    self.scheduler_types[k] = list(set(curr_list))
            else:
                self.scheduler_types = other.scheduler_types

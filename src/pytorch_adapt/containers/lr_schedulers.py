from ..utils import common_functions as c_f
from .base_container import BaseContainer


class LRSchedulers(BaseContainer):
    """
    A container for optimizer learning rate schedulers.
    """

    def __init__(self, store, scheduler_types=None, **kwargs):
        """
        Arguments:
            store: See [```BaseContainer```][pytorch_adapt.containers.base_container.BaseContainer]
            scheduler_types: A dictionary mapping from
                scheduler type (```"per_step"``` or ```"per_epoch"```)
                to a list of object names. If ```None```, then all
                schedulers are assumed to be ```"per_step"```
            **kwargs: [```BaseContainer```][pytorch_adapt.containers.base_container.BaseContainer]
                keyword arguments.
        """
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

    def step(self, scheduler_type: str):
        """
        Step the lr schedulers of the specified type.
        Arguments:
            scheduler_type: ```"per_step"``` or ```"per_epoch"```
        """
        for v in self.filter_by_scheduler_type(scheduler_type):
            v.step()

    def filter_by_scheduler_type(self, x):
        if self.scheduler_types is not None:
            return [v for k, v in self.items() if k in self.scheduler_types[x]]
        elif x == "per_step":
            return self.values()
        elif x == "per_epoch":
            return []
        else:
            raise ValueError(
                f"scheduler types are 'per_step' or 'per_epoch', but input is '{x}'"
            )

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

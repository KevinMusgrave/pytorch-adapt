from .base_container import BaseContainer
from .lr_schedulers import LRSchedulers
from .misc import Misc
from .models import Models
from .optimizers import Optimizers


def get_container(k):
    if k == "models":
        return Models({})
    elif k == "optimizers":
        return Optimizers({})
    elif k == "lr_schedulers":
        return LRSchedulers({})
    elif k == "misc":
        return Misc({})
    else:
        return BaseContainer({})


class MultipleContainers(BaseContainer):
    def __init__(self, **kwargs):
        self.store = kwargs

    def merge(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, BaseContainer):
                if k in self:
                    self[k].merge(v)
                else:
                    self[k] = v
            else:
                if k not in self:
                    self[k] = get_container(k)

    def create(self):
        self["models"].create()
        self["optimizers"].create_with(self["models"])
        self["lr_schedulers"].create_with(self["optimizers"])
        self["misc"].create()

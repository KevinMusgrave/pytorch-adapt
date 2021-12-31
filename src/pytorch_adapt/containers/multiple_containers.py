from typing import Union

from ..utils import common_functions as c_f
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
    """
    Contains other containers and initializes them.
    """

    def __init__(self, **kwargs):
        self.store = kwargs

    def merge(self, **kwargs: Union[BaseContainer, None]):
        """
        Merges the input containers into any existing sub-containers.
        """
        for k, v in kwargs.items():
            if isinstance(v, BaseContainer):
                if k in self:
                    self[k].merge(v)
                else:
                    self[k] = v
            elif v is None:
                if k not in self:
                    self[k] = get_container(k)
            else:
                raise TypeError(
                    f"Input to {c_f.cls_name(self)}.merge must be BaseContainer or None"
                )

    def create(self):
        """
        Calls [```.create()```][pytorch_adapt.containers.BaseContainer.create]
        or [```.create_with()```][pytorch_adapt.containers.BaseContainer.create_with]
        on sub-containers.

        - Optimizers are created with models as input.
        - LR schedulers are created with optimizers as input.
        """
        self["models"].create()
        self["optimizers"].create_with(self["models"])
        self["lr_schedulers"].create_with(self["optimizers"])
        self["misc"].create()

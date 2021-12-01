import inspect
from abc import ABC, abstractmethod
from typing import Dict, List

from ..utils import common_functions as c_f


class BaseValidator(ABC):
    def __init__(self, key_map: Dict[str, str] = None):
        """
        Arguments:
            key_map: A mapping from ```<new_split_names>``` to
                ```<original_split_names>```. For example,
                [```AccuracyValidator```][pytorch_adapt.validators.AccuracyValidator]
                expects ```src_val``` by default. When used with one of the
                [```frameworks```](../frameworks/index.md), this default
                indicates that data related to the ```src_val``` split should be retrieved.
                If you instead want to compute accuracy for the ```src_train``` split,
                you would set the ```key_map``` to ```{"src_train": "src_val"}```.
        """
        self.key_map = c_f.default(key_map, {})

    def _required_data(self):
        args = inspect.getfullargspec(self.compute_score).args
        args.remove("self")
        return args

    @property
    def required_data(self) -> List[str]:
        """
        Returns:
            A list of dataset split names.
        """
        output = set(self._required_data()) - set(self.key_map.values())
        output = list(output)
        for k, v in self.key_map.items():
            output.append(k)
        return output

    @abstractmethod
    def compute_score(self):
        pass

    def score(self, **kwargs):
        kwargs = self.kwargs_check(kwargs)
        return self.compute_score(**kwargs)

    def kwargs_check(self, kwargs):
        if kwargs.keys() != set(self.required_data):
            raise ValueError(
                f"Input to compute_score has keys = {kwargs.keys()} but should have keys {self.required_data}"
            )
        return c_f.map_keys(kwargs, self.key_map)

    def __repr__(self):
        return c_f.nice_repr(self, self.extra_repr(), {})

    def extra_repr(self):
        return c_f.extra_repr(self, ["required_data"])

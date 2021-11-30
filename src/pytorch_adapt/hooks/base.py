from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from ..utils import common_functions as c_f


class BaseHook(ABC):
    """All hooks extend ```BaseHook```"""

    def __init__(
        self,
        loss_prefix: str = "",
        loss_suffix: str = "",
        out_prefix: str = "",
        out_suffix: str = "",
        key_map: Dict[str, str] = None,
    ):
        """
        Arguments:
            loss_prefix: prepended to all new loss keys
            loss_suffix: appended to all new loss keys
            out_prefix: prepended to all new output keys
            out_suffix: appended to all new output keys
            key_map: a mapping from ```input_key``` to ```new_key```.
                For example, if key_map = {"A": "B"}, and the input dict to ```__call__``` is {"A": 5},
                then the input will be converted to {"B": 5} before being consumed. Before exiting ```__call__```,
                the mapping is undone so the input context is preserved.
                In other words, {"B": 5} will be converted back to {"A": 5}.
        """
        if any(
            not isinstance(x, str)
            for x in [loss_prefix, loss_suffix, out_prefix, out_suffix]
        ):
            raise TypeError("loss prefix/suffix and out prefix/suffix must be strings")
        self.loss_prefix = loss_prefix
        self.loss_suffix = loss_suffix
        self.out_prefix = out_prefix
        self.out_suffix = out_suffix
        self.key_map = c_f.default(key_map, {})
        self.in_keys = []

    def __call__(self, losses, inputs):
        try:
            inputs = c_f.map_keys(inputs, self.key_map)
            x = self.call(losses, inputs)
            if isinstance(x, (bool, np.bool_)):
                return x
            elif isinstance(x, tuple):
                losses, outputs = x
                outputs = replace_mapped_keys(outputs, self.key_map)
                inputs = replace_mapped_keys(inputs, self.key_map)
                losses = wrap_keys(losses, self.loss_prefix, self.loss_suffix)
                outputs = wrap_keys(outputs, self.out_prefix, self.out_suffix)
                self.check_losses_and_outputs(losses, outputs, inputs)
                return losses, outputs
            else:
                raise TypeError(
                    f"Output is of type {type(x)}, but should be bool or tuple"
                )
        except Exception as e:
            if not isinstance(e, KeyError):
                c_f.append_error_message(e, self.str_for_error_msg(n=1))
            raise

    @abstractmethod
    def call(
        self, losses: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], bool]:
        """
        This must be implemented by the child class
        Arguments:
            losses: previously computed losses
            inputs: holds everything else: tensors, models etc.
        Returns:
            Either a tuple of (losses, outputs) that will be merged with the input context,
            or a boolean
        """
        pass

    @abstractmethod
    def _loss_keys(self) -> List[str]:
        """
        This must be implemented by the child class
        Returns:
            The names of the losses that will be added to the context.
        """
        pass

    @property
    def loss_keys(self):
        return list(
            set(wrap_keys(self._loss_keys(), self.loss_prefix, self.loss_suffix))
        )

    @abstractmethod
    def _out_keys(self) -> List[str]:
        """
        This must be implemented by the child class
        Returns:
            The names of the outputs that will be added to the context.
        """
        pass

    @property
    def out_keys(self):
        x = replace_mapped_keys(self._out_keys(), self.key_map)
        return list(set(wrap_keys(x, self.out_prefix, self.out_suffix)))

    def set_in_keys(self, in_keys):
        self.in_keys = in_keys

    def __repr__(self):
        return c_f.nice_repr(self, self.extra_repr(), self.children_repr())

    def extra_repr(self):
        return ""

    def children_repr(self):
        all_hooks = c_f.attrs_of_type(self, BaseHook)
        all_modules = c_f.attrs_of_type(self, torch.nn.Module)
        return c_f.assert_dicts_are_disjoint(all_hooks, all_modules)

    def str_for_error_msg(self, x=None, n=None):
        e = str(self if x is None else x)
        if n is not None:
            e = "\n".join(e.split("\n")[:n])
            e += "\n...\n"
        return f"\nERROR occuring in:\n{e}"

    def check_losses_and_outputs(self, losses, outputs, inputs):
        check_keys_are_present(self, self.loss_keys, [losses], "loss_keys", "losses")
        check_keys_are_present(
            self, self.out_keys, [inputs, outputs], "out_keys", "inputs or outputs"
        )
        check_keys_are_present(self, losses, self.loss_keys, "loss_keys", "losses")
        check_keys_are_present(self, outputs, self.out_keys, "outputs", "out_keys")


def wrap_keys(x, prefix, suffix):
    if prefix == "" and suffix == "":
        return x
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[f"{prefix}{k}{suffix}"] = v
        return new_x
    elif c_f.is_list_or_tuple(x):
        return [f"{prefix}{y}{suffix}" for y in x]
    else:
        raise TypeError("input to wrap_keys should be dict, list, or tuple")


def replace_mapped_keys(outputs, key_map):
    if len(key_map) == 0:
        return outputs
    reverse_key_map = {v: k for k, v in key_map.items()}
    if isinstance(outputs, dict):
        return {
            c_f.map_keys_substrings(k, reverse_key_map): v for k, v in outputs.items()
        }
    elif c_f.is_list_or_tuple(outputs):
        return [c_f.map_keys_substrings(k, reverse_key_map) for k in outputs]


def check_keys_are_present(cls, x, y, x_name, y_name):
    set_x = c_f.to_set(x)
    set_y = c_f.to_set(y)
    diff = set_x - set_y
    if len(diff) > 0:
        msg = f"keys {diff} are in {x_name} but are not present in {y_name} of {c_f.cls_name(cls)}"
        raise KeyError(msg)


class BaseConditionHook(BaseHook):
    """The base class for hooks that return a boolean"""

    def _loss_keys(self):
        """"""
        return []

    def _out_keys(self):
        """"""
        return []


class BaseWrapperHook(BaseHook):
    """A simple wrapper for calling ```self.hook```,
    which should be defined in the child's ```__init__``` function."""

    def call(self, *args, **kwargs):
        """"""
        return self.hook(*args, **kwargs)

    def _loss_keys(self):
        """"""
        return self.hook.loss_keys

    def _out_keys(self):
        """"""
        return self.hook.out_keys

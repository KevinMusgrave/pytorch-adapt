import re
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch

from ..layers import EntropyWeights
from ..utils import common_functions as c_f
from .base import BaseHook, BaseWrapperHook
from .features import FeaturesAndLogitsHook


class BaseReducer(BaseHook, ABC):
    def __init__(self, apply_to=None, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.apply_to = apply_to
        self.default_reducer = default_reducer
        self.curr_loss_keys = []

    def call(self, losses, inputs):
        self.curr_loss_keys = list(losses.keys())
        apply_to = self.get_keys_to_apply_to(losses)
        losses, outputs = self.call_reducer(losses, inputs, apply_to)
        if self.default_reducer:
            c_f.assert_dicts_are_disjoint(inputs, outputs)
            losses, new_outputs = self.default_reducer(losses, {**inputs, **outputs})
            outputs.update(new_outputs)
        if losses.keys() != set(self.curr_loss_keys):
            raise ValueError(
                "Loss dict returned by reducer should have same keys as input loss dict"
            )
        return losses, outputs

    @abstractmethod
    def call_reducer(self, losses, inputs, apply_to):
        pass

    def _loss_keys(self):
        return self.curr_loss_keys

    def get_keys_to_apply_to(self, losses):
        apply_to = self.apply_to
        if apply_to is None:
            apply_to = [k for k, v in losses.items() if not c_f.len_one_tensor(v)]
        elif len(set(apply_to) - set(self.curr_loss_keys)) > 0:
            raise ValueError(
                f"self.apply_to ({self.apply_to}) must be a subset of losses.keys() ({losses.keys()})"
            )
        return apply_to

    def extra_repr(self):
        return c_f.extra_repr(self, ["apply_to"])


class MultipleReducers(BaseHook):
    def __init__(self, reducers, **kwargs):
        super().__init__(**kwargs)
        self.reducers = reducers

    def call(self, losses, inputs):
        for r in self.reducers:
            losses, inputs = r(losses, inputs)
        return losses, inputs

    def _loss_keys(self):
        return c_f.join_lists([r.loss_keys for r in self.reducers])

    def _out_keys(self):
        return c_f.join_lists([r.out_keys for r in self.reducers])


class EntropyReducer(BaseReducer):
    def __init__(
        self,
        f_hook=None,
        domains=None,
        entropy_weights_fn=None,
        detach_weights=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        src_regex = "^{0}_|_{0}$|_{0}_|^{0}$".format("src")
        target_regex = "^{0}_|_{0}$|_{0}_|^{0}$".format("target")
        self.src_regex = re.compile(src_regex)
        self.target_regex = re.compile(target_regex)
        self.entropy_weights_fn = c_f.default(entropy_weights_fn, EntropyWeights, {})
        self.f_hook = c_f.default(
            f_hook,
            FeaturesAndLogitsHook,
            {
                "detach_features": detach_weights,
                "detach_logits": detach_weights,
                "domains": domains,
            },
        )
        self.context = torch.no_grad() if detach_weights else nullcontext()

    def call_reducer(self, losses, inputs, apply_to):
        outputs = self.f_hook(losses, inputs)[1]
        for k in apply_to:
            if self.src_regex.search(k):
                domain = "src"
            elif self.target_regex.search(k):
                domain = "target"
            else:
                raise ValueError
            with self.context:
                search_str = c_f.filter(self.f_hook.out_keys, "_logits", [f"^{domain}"])
                [logits] = c_f.extract([outputs, inputs], search_str)
                weights = self.entropy_weights_fn(logits)
            losses[k] = torch.mean(weights * losses[k])

        return losses, outputs

    def _out_keys(self):
        return self.f_hook.out_keys


class MeanReducer(BaseReducer):
    def call_reducer(self, losses, inputs, apply_to):
        for k in apply_to:
            losses[k] = torch.mean(losses[k])
        return losses, {}

    def _out_keys(self):
        return []

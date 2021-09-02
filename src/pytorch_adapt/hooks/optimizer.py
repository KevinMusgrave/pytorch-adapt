from ..utils import common_functions as c_f
from ..weighters import MeanWeighter
from .base import BaseHook, BaseWrapperHook
from .reducers import MeanReducer


class OptimizerHook(BaseHook):
    # optimizers is a list of optimizers
    def __init__(self, hook, optimizers, weighter=None, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.hook = hook
        self.optimizers = optimizers
        self.weighter = c_f.default(weighter, MeanWeighter, {})
        self.reducer = c_f.default(reducer, MeanReducer, {})
        self.loss_components = {}

    def call(self, losses, inputs):
        losses, outputs = self.hook(losses, inputs)
        c_f.assert_dicts_are_disjoint(inputs, outputs)
        losses, new_outputs = self.reducer(losses, {**inputs, **outputs})
        outputs.update(new_outputs)
        loss, self.loss_components = self.weighter(losses)
        c_f.zero_back_step(loss, self.optimizers)
        return {}, outputs

    def _loss_keys(self):
        return []

    def _out_keys(self):
        return c_f.join_lists([self.hook.out_keys, self.reducer.out_keys])

    def extra_repr(self):
        return c_f.extra_repr(self, ["weighter"])


class SummaryHook(BaseHook):
    # optimizers is a dict of optimizer hooks
    def __init__(self, optimizers, **kwargs):
        super().__init__(**kwargs)
        self.optimizers = optimizers

    def call(self, losses, inputs):
        losses = {}
        for k, v in self.optimizers.items():
            losses[k] = v.loss_components
        return losses, {}

    def _loss_keys(self):
        return list(self.optimizers.keys())

    def _out_keys(self):
        return []

from typing import Dict, List

import torch

from ..utils import common_functions as c_f
from ..weighters import BaseWeighter, MeanWeighter
from .base import BaseHook
from .reducers import BaseReducer, MeanReducer


class OptimizerHook(BaseHook):
    """
    1. Executes the wrapped hook
    2. Zeros all gradients
    3. Backpropagates the loss
    4. Steps the optimizer
    """

    def __init__(
        self,
        hook: BaseHook,
        optimizers: List[torch.optim.Optimizer],
        weighter: BaseWeighter = None,
        reducer: BaseReducer = None,
        **kwargs
    ):
        """
        Arguments:
            hook: the hook that computes the losses
            optimizers: a list of optimizers that will be used
                to update model weights
            weighter: weights the returned losses and outputs a
                single value on which ```.backward()``` is called.
                If ```None```, then it defaults to
                [```MeanWeighter```][pytorch_adapt.weighters.mean_weighter.MeanWeighter].
            reducer: a hook that reduces any unreduced losses to a single value.
                If ```None```, then it defaults to
                [```MeanReducer```][pytorch_adapt.hooks.reducers.MeanReducer].
        """
        super().__init__(**kwargs)
        self.hook = hook
        self.optimizers = optimizers
        self.weighter = c_f.default(weighter, MeanWeighter, {})
        self.reducer = c_f.default(reducer, MeanReducer, {})
        self.loss_components = {}

    def call(self, losses, inputs):
        """"""
        losses, outputs = self.hook(losses, inputs)
        combined = c_f.assert_dicts_are_disjoint(inputs, outputs)
        losses, new_outputs = self.reducer(losses, combined)
        outputs.update(new_outputs)
        loss, self.loss_components = self.weighter(losses)
        optimizers = self.optimizers
        if isinstance(optimizers[0], str):
            optimizers = c_f.extract(inputs, optimizers)
        c_f.zero_back_step(loss, optimizers, inputs.get("custom_backward"))
        return {}, outputs

    def _loss_keys(self):
        """"""
        return []

    def _out_keys(self):
        """"""
        return c_f.join_lists([self.hook.out_keys, self.reducer.out_keys])

    def extra_repr(self):
        return c_f.extra_repr(self, ["optimizers", "weighter"])


class SummaryHook(BaseHook):
    """
    Repackages losses into a dictionary format useful for logging.
    This should be used only at the very end of each
    iteration, i.e. it should be the last sub-hook
    in a [ChainHook][pytorch_adapt.hooks.utils.ChainHook].
    """

    def __init__(self, optimizers: Dict[str, OptimizerHook], **kwargs):
        """
        Arguments:
            optimizers: A dictionary of optimizer hooks.
                The losses computed inside these hooks
                will be packaged into nested dictionaries.
        """
        super().__init__(**kwargs)
        self.optimizers = optimizers

    def call(self, losses, inputs):
        """"""
        losses = {}
        for k, v in self.optimizers.items():
            losses[k] = v.loss_components
        return losses, {}

    def _loss_keys(self):
        """"""
        return list(self.optimizers.keys())

    def _out_keys(self):
        """"""
        return []

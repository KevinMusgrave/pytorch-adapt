from typing import Callable, List

import torch

from ..utils import common_functions as c_f
from .base import BaseHook, BaseWrapperHook
from .features import (
    FeaturesAndLogitsHook,
    FeaturesChainHook,
    FeaturesHook,
    FrozenModelHook,
    LogitsHook,
)
from .optimizer import OptimizerHook, SummaryHook
from .utils import ApplyFnHook, ChainHook, OnlyNewOutputsHook


class SoftmaxHook(ApplyFnHook):
    """
    Applies ```torch.nn.Softmax(dim=1)``` to the
    specified inputs.

    Extends [```ApplyFnHook```][pytorch_adapt.hooks.utils.ApplyFnHook]
    """

    def __init__(self, **kwargs):
        super().__init__(fn=torch.nn.Softmax(dim=1), **kwargs)


class SoftmaxLocallyHook(BaseWrapperHook):
    """
    Applies ```torch.nn.Softmax(dim=1)``` to the
    specifieid inputs, which are overwritten, but
    only inside this hook.
    """

    def __init__(self, apply_to: List[str], *hooks: BaseHook, **kwargs):
        """
        Arguments:
            apply_to: list of names of tensors that softmax
                will be applied to.
            hooks: the hooks that will receive the softmaxed
                tensors.
        """
        super().__init__(**kwargs)
        s_hook = SoftmaxHook(apply_to=apply_to)
        self.hook = OnlyNewOutputsHook(ChainHook(s_hook, *hooks, overwrite=True))


class CLossHook(BaseWrapperHook):
    """
    Computes a classification loss on the specified tensors.
    The default setting is to compute the cross entropy loss
    of the source domain logits.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        detach_features: bool = False,
        f_hook: BaseHook = None,
        **kwargs,
    ):
        """
        Arguments:
            loss_fn: The classification loss function. If ```None```,
                it defaults to ```torch.nn.CrossEntropyLoss```.
            detach_features: Whether or not to detach the features,
                from which logits are computed.
            f_hook: The hook for computing logits.
        """

        super().__init__(**kwargs)
        self.loss_fn = c_f.default(
            loss_fn, torch.nn.CrossEntropyLoss, {"reduction": "none"}
        )
        self.hook = c_f.default(
            f_hook,
            FeaturesAndLogitsHook,
            {"domains": ["src"], "detach_features": detach_features},
        )

    def call(self, losses, inputs):
        """"""
        outputs = self.hook(losses, inputs)[1]
        [src_logits] = c_f.extract(
            [outputs, inputs], c_f.filter(self.hook.out_keys, "_logits$")
        )
        loss = self.loss_fn(src_logits, inputs["src_labels"])
        return {self._loss_keys()[0]: loss}, outputs

    def _loss_keys(self):
        """"""
        return ["c_loss"]


class ClassifierHook(BaseWrapperHook):
    """
    This computes the classification loss and also
    optimizes the models.
    """

    def __init__(
        self,
        opts,
        weighter=None,
        reducer=None,
        loss_fn=None,
        f_hook=None,
        detach_features=False,
        pre=None,
        post=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        [pre, post] = c_f.many_default([pre, post], [[], []])
        hook = CLossHook(loss_fn, detach_features, f_hook)
        hook = ChainHook(*pre, hook, *post)
        hook = OptimizerHook(hook, opts, weighter, reducer)
        s_hook = SummaryHook({"total_loss": hook})
        self.hook = ChainHook(hook, s_hook)


class FinetunerHook(ClassifierHook):
    """
    This is the same as
    [```ClassifierHook```][pytorch_adapt.hooks.classification.ClassifierHook],
    but it freezes the generator model ("G").
    """

    def __init__(self, **kwargs):
        f_hook = FrozenModelHook(FeaturesHook(detach=True, domains=["src"]), "G")
        f_hook = FeaturesChainHook(f_hook, LogitsHook(domains=["src"]))
        super().__init__(f_hook=f_hook, **kwargs)

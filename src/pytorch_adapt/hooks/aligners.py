from typing import Callable, List

import torch

from ..layers import MMDLoss
from ..utils import common_functions as c_f
from .base import BaseHook, BaseWrapperHook
from .classification import CLossHook, SoftmaxLocallyHook
from .features import FeaturesAndLogitsHook, FeaturesHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import ChainHook


class AlignerHook(BaseWrapperHook):
    """
    Computes an alignment loss (e.g MMD) based on features
    from two domains.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        hook: BaseHook = None,
        layer: str = "features",
        **kwargs,
    ):
        """
        Arguments:
            loss_fn: a function that computes a distance
                between two tensors. If ```None```,
                it defaults to [```MMDLoss```][pytorch_adapt.layers.mmd_loss.MMDLoss].
            hook: the hook for computing features
            layer: the layer for which the loss is computed. Must be
                either ```"features"``` or ```"logits"```.
        """

        super().__init__(**kwargs)
        self.loss_fn = c_f.default(loss_fn, MMDLoss, {})
        if layer == "features":
            default_hook = FeaturesHook
        elif layer == "logits":
            default_hook = FeaturesAndLogitsHook
        else:
            raise ValueError("AlignerHook layer must be 'features' or 'logits'")
        self.hook = c_f.default(hook, default_hook, {})
        self.layer = layer

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        strs = c_f.filter(self.hook.out_keys, f"_{self.layer}$", ["^src", "^target"])
        [src, target] = c_f.extract([outputs, inputs], strs)
        confusion_loss = self.loss_fn(src, target)
        return {self._loss_keys()[0]: confusion_loss}, outputs

    def _loss_keys(self):
        return [f"{self.layer}_confusion_loss"]


class JointAlignerHook(BaseWrapperHook):
    """
    Computes a joint alignment loss (e.g Joint MMD) based on
    multiple features from two domains.

    The default setting is to use the features and logits
    from the source and target domains.
    """

    def __init__(
        self,
        loss_fn: Callable[
            [List[torch.Tensor], List[torch.Tensor]], torch.Tensor
        ] = None,
        hook: BaseHook = None,
        **kwargs,
    ):
        """
        Arguments:
            loss_fn: a function that computes a distance
                between two **lists** of tensors. If ```None```,
                it defaults to [```MMDLoss```][pytorch_adapt.layers.mmd_loss.MMDLoss].
            hook: the hook for computing features and logits
        """
        super().__init__(**kwargs)
        self.loss_fn = c_f.default(loss_fn, MMDLoss, {})
        self.hook = c_f.default(hook, FeaturesAndLogitsHook, {})

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        src = self.get_all_domain_features(inputs, outputs, "src")
        target = self.get_all_domain_features(inputs, outputs, "target")
        confusion_loss = self.loss_fn(src, target)
        return {self._loss_keys()[0]: confusion_loss}, outputs

    def _loss_keys(self):
        return ["joint_confusion_loss"]

    def get_all_domain_features(self, inputs, outputs, domain):
        return c_f.extract(
            [outputs, inputs], sorted(c_f.filter(self.hook.out_keys, f"^{domain}"))
        )


class FeaturesLogitsAlignerHook(BaseWrapperHook):
    """
    This chains together an
    [```AlignerHook```][pytorch_adapt.hooks.aligners.AlignerHook] for
    ```"features"``` followed by an ```AlignerHook``` for ```"logits"```.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        **kwargs,
    ):
        """
        Arguments:
            loss_fn: The loss used by both aligner hooks.
        """
        super().__init__(**kwargs)
        loss_fn = c_f.default(loss_fn, MMDLoss, {})
        a1_hook = AlignerHook(loss_fn, layer="features")
        a2_hook = AlignerHook(loss_fn, layer="logits")
        self.hook = ChainHook(a1_hook, a2_hook)


class ManyAlignerHook(BaseWrapperHook):
    def __init__(
        self,
        loss_fn=None,
        aligner_hook=None,
        features_hook=None,
        softmax=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        loss_fn = c_f.default(loss_fn, MMDLoss, {})
        features_hook = c_f.default(features_hook, FeaturesAndLogitsHook, {})
        aligner_hook = c_f.default(
            aligner_hook, FeaturesLogitsAlignerHook, {"loss_fn": loss_fn}
        )
        if softmax:
            apply_to = c_f.filter(features_hook.out_keys, "_logits$")
            aligner_hook = SoftmaxLocallyHook(apply_to, aligner_hook)
        self.hook = ChainHook(features_hook, aligner_hook)


class AlignerPlusCHook(BaseWrapperHook):
    """
    Computes an alignment loss plus a classification loss,
    and then optimizes the models.
    """

    def __init__(
        self,
        opts,
        weighter=None,
        reducer=None,
        loss_fn=None,
        aligner_hook=None,
        pre=None,
        post=None,
        softmax=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        [pre, post] = c_f.many_default([pre, post], [[], []])
        aligner_hook = ManyAlignerHook(
            loss_fn=loss_fn, aligner_hook=aligner_hook, softmax=softmax
        )
        hook = ChainHook(*pre, aligner_hook, CLossHook(), *post)
        hook = OptimizerHook(hook, opts, weighter, reducer)
        s_hook = SummaryHook({"total_loss": hook})
        self.hook = ChainHook(hook, s_hook)

from ..layers import MMDLoss
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import CLossHook, SoftmaxLocallyHook
from .features import FeaturesAndLogitsHook, FeaturesHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import ChainHook


class AlignerHook(BaseWrapperHook):
    def __init__(self, loss_fn=None, hook=None, layer="features", **kwargs):
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
    def __init__(self, loss_fn=None, hook=None, **kwargs):
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
    def __init__(self, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        loss_fn = c_f.default(loss_fn, MMDLoss, {})
        features_hook = FeaturesAndLogitsHook()
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

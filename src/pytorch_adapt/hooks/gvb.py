from ..layers import AbsLoss
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import SoftmaxLocallyHook
from .dann import (
    DANNHook,
    GradientReversalThenEntropyReducer,
    SoftmaxGradientReversalHook,
)
from .domain import FeaturesForDomainLossHook
from .features import (
    BaseFeaturesHook,
    FeaturesAndLogitsHook,
    FeaturesChainHook,
    FeaturesHook,
)
from .gan import GANHook
from .utils import ChainHook, MultiplierHook


class BridgeAndLogitsHook(BaseFeaturesHook):
    def add_if_new(
        self, outputs, full_key, output_vals, inputs, model_name, in_keys, domain
    ):
        # assume model takes 2 inputs: model(x, return_bridge)
        c_f.add_if_new(
            outputs,
            full_key,
            output_vals,
            inputs,
            model_name,
            in_keys,
            other_args=[True],
        )


class GBridgeAndLogitsHook(BridgeAndLogitsHook):
    def __init__(
        self,
        model_name="C",
        in_suffixes=None,
        out_suffixes=None,
        **kwargs,
    ):
        in_suffixes = c_f.default(in_suffixes, ["_imgs_features"])
        out_suffixes = c_f.default(out_suffixes, ["_logits", "_gbridge"])

        super().__init__(
            model_name=model_name,
            in_suffixes=in_suffixes,
            out_suffixes=out_suffixes,
            **kwargs,
        )


class DBridgeAndLogitsHook(BridgeAndLogitsHook):
    def __init__(
        self,
        model_name="D",
        in_suffixes=None,
        out_suffixes=None,
        **kwargs,
    ):
        in_suffixes = c_f.default(in_suffixes, ["_imgs_features_logits"])
        out_suffixes = c_f.default(out_suffixes, ["_dlogits", "_dbridge"])
        super().__init__(
            model_name=model_name,
            in_suffixes=in_suffixes,
            out_suffixes=out_suffixes,
            **kwargs,
        )


class FeaturesLogitsAndGBridge(FeaturesChainHook):
    def __init__(self, **kwargs):
        hooks = [FeaturesHook(), GBridgeAndLogitsHook()]
        super().__init__(*hooks, **kwargs)


class FeaturesLogitsAndDBridge(BaseWrapperHook):
    def __init__(self, detach_features=False, **kwargs):
        super().__init__(**kwargs)
        self.hook = FeaturesAndLogitsHook(
            detach_features=detach_features,
            detach_logits=detach_features,
            other_hooks=[DBridgeAndLogitsHook()],
        )


class BridgeLossHook(BaseWrapperHook):
    def __init__(self, hook, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.hook = hook
        self.loss_fn = c_f.default(loss_fn, AbsLoss, {})

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        strs = c_f.filter(self.hook.out_keys, f"_[a-z]bridge$", ["^src", "^target"])
        [src_bridge, target_bridge] = c_f.extract([outputs, inputs], strs)
        return {
            f"src_bridge_loss": self.loss_fn(src_bridge),
            f"target_bridge_loss": self.loss_fn(target_bridge),
        }, outputs

    def _loss_keys(self):
        return [f"src_bridge_loss", f"target_bridge_loss"]


class GBridgeLossHook(BridgeLossHook):
    def __init__(self, hook=None, loss_prefix="g_", **kwargs):
        hook = c_f.default(hook, FeaturesLogitsAndGBridge, {})
        super().__init__(hook=hook, loss_prefix=loss_prefix, **kwargs)


class DBridgeLossHook(BridgeLossHook):
    def __init__(self, hook=None, loss_prefix="d_", detach_features=False, **kwargs):
        hook = c_f.default(
            hook, FeaturesLogitsAndDBridge, {"detach_features": detach_features}
        )
        super().__init__(hook=hook, loss_prefix=loss_prefix, **kwargs)


class GVBHook(DANNHook):
    """
    Implementation of
    [Gradually Vanishing Bridge for Adversarial Domain Adaptation](https://arxiv.org/abs/2003.13183)
    """

    def __init__(
        self, gradient_reversal_weight=1, pre=None, pre_d=None, pre_g=None, **kwargs
    ):
        # f_hook and d_hook are used inside DomainLossHook
        f_hook = FeaturesForDomainLossHook(use_logits=True)
        d_hook = DBridgeAndLogitsHook()
        apply_to = c_f.filter(f_hook.out_keys, "_logits$")
        gradient_reversal = SoftmaxGradientReversalHook(
            weight=gradient_reversal_weight, apply_to=apply_to
        )
        [pre, pre_d, pre_g] = c_f.many_default([pre, pre_d, pre_g], [[], [], []])
        pre += [FeaturesLogitsAndGBridge()]
        pre_d += [DBridgeLossHook()]
        pre_g += [GBridgeLossHook()]

        super().__init__(
            pre=pre,
            pre_d=pre_d,
            pre_g=pre_g,
            gradient_reversal=gradient_reversal,
            f_hook=f_hook,
            d_hook=d_hook,
            d_hook_allowed="_dlogits$|_dbridge$",
            **kwargs,
        )


class GVBEHook(GVBHook):
    def __init__(
        self, detach_entropy_reducer=True, gradient_reversal_weight=1, **kwargs
    ):
        reducer = GradientReversalThenEntropyReducer(
            detach_entropy_reducer, gradient_reversal_weight
        )
        super().__init__(
            reducer=reducer, gradient_reversal_weight=gradient_reversal_weight, **kwargs
        )


class DBridgeLossWithSoftmaxHook(BaseWrapperHook):
    def __init__(self, detach_features=False, **kwargs):
        super().__init__(**kwargs)
        gbridge_hook = FeaturesLogitsAndGBridge()
        self.hook = ChainHook(
            gbridge_hook,
            SoftmaxLocallyHook(
                c_f.filter(gbridge_hook.out_keys, "_logits"),
                DBridgeLossHook(detach_features=detach_features),
            ),
        )


class GVBGANHook(GANHook):
    def __init__(self, pre=None, pre_d=None, pre_g=None, **kwargs):
        [pre, pre_d, pre_g] = c_f.many_default([pre, pre_d, pre_g], [[], [], []])
        disc_d_hook = DBridgeAndLogitsHook()
        gen_d_hook = DBridgeAndLogitsHook()
        pre_d += [
            FeaturesLogitsAndGBridge(),
            DBridgeLossWithSoftmaxHook(detach_features=True),
        ]
        pre_g += [MultiplierHook(DBridgeLossWithSoftmaxHook(), -1), GBridgeLossHook()]
        super().__init__(
            pre_d=pre_d,
            pre_g=pre_g,
            use_logits=True,
            disc_d_hook=disc_d_hook,
            gen_d_hook=gen_d_hook,
            **kwargs,
        )

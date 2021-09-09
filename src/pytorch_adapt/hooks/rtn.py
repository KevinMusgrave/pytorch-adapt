from ..layers import MMDLoss
from ..utils import common_functions as c_f
from .aligners import AlignerHook
from .base import BaseWrapperHook
from .classification import CLossHook
from .features import BaseFeaturesHook, CombinedFeaturesHook, FeaturesAndLogitsHook
from .losses import TargetEntropyHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import ChainHook


class ResidualHook(BaseFeaturesHook):
    def __init__(
        self,
        in_suffixes=None,
        out_suffixes=None,
        **kwargs,
    ):
        in_suffixes = c_f.default(in_suffixes, ["_imgs_features_logits"])
        out_suffixes = c_f.default(out_suffixes, ["_plus_residual"])
        super().__init__(
            model_name="residual_model",
            in_suffixes=in_suffixes,
            out_suffixes=out_suffixes,
            **kwargs,
        )


def rtn_aligner_key_map(fc_hook):
    strs = c_f.filter(fc_hook.out_keys, "", ["^src", "^target"])
    return {k: v for k, v in zip(strs, ["src_imgs_features", "target_imgs_features"])}


class RTNAlignerHook(BaseWrapperHook):
    def __init__(self, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        loss_fn = c_f.default(loss_fn, MMDLoss, {})
        fl_hook = FeaturesAndLogitsHook()
        fc_hook = CombinedFeaturesHook()
        key_map = rtn_aligner_key_map(fc_hook)
        aligner_hook = AlignerHook(loss_fn=loss_fn, layer="features", key_map=key_map)
        self.hook = ChainHook(fl_hook, fc_hook, aligner_hook)


class RTNLogitsHook(BaseWrapperHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fl_hook = FeaturesAndLogitsHook()
        entropy_hook = TargetEntropyHook()
        residual_hook = ResidualHook(domains=["src"])
        closs_hook = CLossHook(
            key_map={residual_hook.out_keys[0]: "src_imgs_features_logits"}
        )
        self.hook = ChainHook(fl_hook, entropy_hook, residual_hook, closs_hook)


class RTNHook(BaseWrapperHook):
    """
    Implementation of
    [Unsupervised Domain Adaptation with Residual Transfer Networks](https://arxiv.org/abs/1602.04433).
    """

    def __init__(
        self,
        opts,
        weighter=None,
        reducer=None,
        pre=None,
        post=None,
        aligner_loss_fn=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        [pre, post] = c_f.many_default([pre, post], [[], []])
        hook = ChainHook(*pre, RTNAlignerHook(aligner_loss_fn), RTNLogitsHook(), *post)
        hook = OptimizerHook(hook, opts, weighter, reducer)
        s_hook = SummaryHook({"total_loss": hook})
        self.hook = ChainHook(hook, s_hook)

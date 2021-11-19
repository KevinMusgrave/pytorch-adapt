from ..layers import EntropyWeights, MaxNormalizer
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import SoftmaxLocallyHook
from .domain import DomainLossHook
from .features import CombinedFeaturesHook, FeaturesAndLogitsHook
from .gan import GANHook
from .reducers import EntropyReducer, MeanReducer
from .utils import ChainHook


def cdan_key_map(fc_hook):
    strs = c_f.filter(fc_hook.out_keys, "", ["^src", "^target"])
    return {k: v for k, v in zip(strs, ["src_imgs_features", "target_imgs_features"])}


def get_entropy_reducer(apply_to, detach_weights):
    return EntropyReducer(
        apply_to=apply_to,
        default_reducer=MeanReducer(),
        entropy_weights_fn=EntropyWeights(normalizer=MaxNormalizer(detach=True)),
        detach_weights=detach_weights,
    )


def get_entropy_reducers_for_gan(detach_entropy_reducer):
    d_reducer = get_entropy_reducer(
        ["d_src_domain_loss", "d_target_domain_loss"],
        detach_weights=True,
    )
    g_reducer = get_entropy_reducer(
        ["g_src_domain_loss", "g_target_domain_loss"],
        detach_weights=detach_entropy_reducer,
    )
    return d_reducer, g_reducer


def get_cdan_features_hooks(softmax):
    f_hook = FeaturesAndLogitsHook()
    fc_hook = CombinedFeaturesHook()
    key_map = cdan_key_map(fc_hook)
    if softmax:
        strs = c_f.filter(f_hook.out_keys, "_features_logits$", ["^src", "^target"])
        fc_hook = SoftmaxLocallyHook(strs, fc_hook)
    return f_hook, fc_hook, key_map


class CDANDomainHook(BaseWrapperHook):
    def __init__(
        self, loss_prefix, detach_features, reverse_labels, softmax=True, **kwargs
    ):
        super().__init__(**kwargs)
        f_hook, fc_hook, key_map = get_cdan_features_hooks(softmax)
        d_hook = DomainLossHook(
            loss_prefix=loss_prefix,
            detach_features=detach_features,
            reverse_labels=reverse_labels,
            key_map=key_map,
        )
        self.hook = ChainHook(f_hook, fc_hook, d_hook)


class CDANDomainHookD(CDANDomainHook):
    def __init__(self, **kwargs):
        super().__init__(
            loss_prefix="d_", detach_features=True, reverse_labels=False, **kwargs
        )


class CDANDomainHookG(CDANDomainHook):
    def __init__(self, **kwargs):
        super().__init__(
            loss_prefix="g_", detach_features=False, reverse_labels=True, **kwargs
        )


class CDANHook(GANHook):
    """
    Implementation of
    [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

    Extends [```GANHook```][pytorch_adapt.hooks.gan.GANHook].
    """

    def __init__(self, softmax=True, **kwargs):
        super().__init__(
            disc_hook=CDANDomainHookD(softmax=softmax),
            gen_hook=CDANDomainHookG(softmax=softmax),
            **kwargs
        )


class CDANEHook(CDANHook):
    def __init__(self, detach_entropy_reducer=True, **kwargs):
        d_reducer, g_reducer = get_entropy_reducers_for_gan(detach_entropy_reducer)
        super().__init__(d_reducer=d_reducer, g_reducer=g_reducer, **kwargs)


class GANEHook(GANHook):
    def __init__(self, detach_entropy_reducer=True, **kwargs):
        d_reducer, g_reducer = get_entropy_reducers_for_gan(detach_entropy_reducer)
        super().__init__(d_reducer=d_reducer, g_reducer=g_reducer, **kwargs)

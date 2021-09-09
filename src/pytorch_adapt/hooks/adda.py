from ..utils import common_functions as c_f
from .conditions import StrongDHook
from .features import FeaturesHook, FeaturesWithGradAndDetachedHook, FrozenModelHook
from .gan import GANHook
from .utils import ChainHook, EmptyHook, FalseHook, TrueHook


class ADDAHook(GANHook):
    """
    Implementation of
    [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464).

    Extends [```GANHook```][pytorch_adapt.hooks.gan.GANHook].
    """

    def __init__(self, threshold: float = 0.6, pre_g=None, post_g=None, **kwargs):
        """
        Arguments:
            threshold: In each training iteration, the generator is only updated
                if the discriminator's accuracy is greater than ```threshold```.
        """
        [pre_g, post_g] = c_f.many_default([pre_g, post_g], [[], []])
        sf_frozen = FrozenModelHook(FeaturesHook(detach=True, domains=["src"]), "G")
        tf_all = FeaturesWithGradAndDetachedHook(model_name="T", domains=["target"])
        pre_d = ChainHook(sf_frozen, tf_all)
        num_pre_g = len(pre_g)
        gen_conditions = [TrueHook() for _ in range(num_pre_g + len(post_g) + 2)]
        # generator condition, classifier condition
        gen_conditions[num_pre_g : num_pre_g + 2] = [
            StrongDHook(threshold),
            FalseHook(),
        ]
        super().__init__(
            pre_d=[pre_d],
            pre_g=pre_g,
            post_g=post_g,
            gen_conditions=gen_conditions,
            gen_domains=["target"],
            c_hook=EmptyHook(),
            **kwargs
        )

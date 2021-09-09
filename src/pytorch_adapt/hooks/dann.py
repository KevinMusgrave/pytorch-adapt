from ..layers import GradientReversal
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import CLossHook, SoftmaxHook
from .domain import DomainLossHook, FeaturesForDomainLossHook
from .features import FeaturesHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import (
    ApplyFnHook,
    AssertHook,
    BaseWrapperHook,
    ChainHook,
    OnlyNewOutputsHook,
)


class GradientReversalHook(ApplyFnHook):
    def __init__(self, **kwargs):
        super().__init__(fn=GradientReversal(), **kwargs)


class SoftmaxGradientReversalHook(BaseWrapperHook):
    def __init__(self, apply_to, **kwargs):
        super().__init__(**kwargs)
        self.hook = ChainHook(
            SoftmaxHook(apply_to=apply_to),
            GradientReversalHook(apply_to=apply_to),
            overwrite=True,
        )


class DANNHook(BaseWrapperHook):
    """
    Implementation of
    [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818).

    This includes the model optimization step.
    """

    def __init__(
        self,
        opts,
        weighter=None,
        reducer=None,
        pre=None,
        pre_d=None,
        post_d=None,
        pre_g=None,
        post_g=None,
        gradient_reversal=None,
        use_logits=False,
        f_hook=None,
        d_hook=None,
        c_hook=None,
        domain_loss_hook=None,
        d_hook_allowed="_dlogits$",
        **kwargs
    ):
        """
        Arguments:
            opts: List of optimizers for updating the models.
            weighter: Weights the losses before backpropagation.
                If ```None``` then it defaults to
                [```MeanWeighter```][pytorch_adapt.weighters.mean_weighter.MeanWeighter]
            reducer: Reduces loss tensors.
                If ```None``` then it defaults to
                [```MeanReducer```][pytorch_adapt.hooks.reducers.MeanReducer]
            pre: List of hooks that will be executed at the very
                beginning of each iteration.
            pre_d: List of hooks that will be executed after
                gradient reversal, but before the domain loss.
            post_d: List of hooks that will be executed after
                gradient reversal, and after the domain loss.
            pre_g: List of hooks that will be executed outside of
                the gradient reversal step, and before the generator
                and classifier loss.
            post_g: List of hooks that will be executed after
                the generator and classifier losses.
            gradient_reversal: Called before all D hooks, including
                ```pre_d```.
            use_logits: If ```True```, then D receives the output of C
                instead of the output of G.
            f_hook: The hook used for computing features and logits.
                If ```None``` then it defaults to
                [```FeaturesForDomainLossHook```][pytorch_adapt.hooks.domain.FeaturesForDomainLossHook]
            d_hook: The hook used for computing discriminator logits.
                If ```None``` then it defaults to
                [```DLogitsHook```][pytorch_adapt.hooks.features.DLogitsHook]
            c_hook: The hook used for computing the classifiers's loss.
                If ```None``` then it defaults to
                [```CLossHook```][pytorch_adapt.hooks.classification.CLossHook]
            domain_loss_hook: The hook used for computing the domain loss.
                If ```None``` then it defaults to
                [```DomainLossHook```][pytorch_adapt.hooks.domain.DomainLossHook].
            d_hook_allowed: A regex string that specifies the allowed
                output names of the discriminator block.
        """
        super().__init__(**kwargs)
        [pre, pre_d, post_d, pre_g, post_g] = c_f.many_default(
            [pre, pre_d, post_d, pre_g, post_g], [[], [], [], [], []]
        )
        f_hook = c_f.default(
            f_hook, FeaturesForDomainLossHook, {"use_logits": use_logits}
        )
        gradient_reversal = c_f.default(
            gradient_reversal, GradientReversalHook, {"apply_to": f_hook.out_keys}
        )
        c_hook = c_f.default(c_hook, CLossHook, {})
        domain_loss_hook = c_f.default(
            domain_loss_hook, DomainLossHook, {"f_hook": f_hook, "d_hook": d_hook}
        )

        disc_hook = AssertHook(
            OnlyNewOutputsHook(
                ChainHook(
                    gradient_reversal,
                    *pre_d,
                    domain_loss_hook,
                    *post_d,
                    overwrite=[1],
                )
            ),
            d_hook_allowed,
        )
        gen_hook = ChainHook(*pre_g, c_hook, *post_g)

        hook = ChainHook(*pre, f_hook, disc_hook, gen_hook)
        hook = OptimizerHook(hook, opts, weighter, reducer)
        s_hook = SummaryHook({"total_loss": hook})
        self.hook = ChainHook(hook, s_hook)


class DANNLogitsHook(DANNHook):
    def __init__(self, **kwargs):
        f_hook = FeaturesForDomainLossHook(use_logits=True)
        super().__init__(f_hook=f_hook, **kwargs)


class DANNSoftmaxLogitsHook(DANNHook):
    def __init__(self, **kwargs):
        f_hook = FeaturesForDomainLossHook(use_logits=True)
        gradient_reversal = SoftmaxGradientReversalHook(apply_to=f_hook.out_keys)
        super().__init__(f_hook=f_hook, gradient_reversal=gradient_reversal, **kwargs)

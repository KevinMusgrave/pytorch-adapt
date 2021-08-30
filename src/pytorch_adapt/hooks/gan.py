from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import CLossHook
from .domain import DomainLossHook, FeaturesForDomainLossHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import AssertHook, ChainHook, OnlyNewOutputsHook


class GANHook(BaseWrapperHook):
    def __init__(
        self,
        d_opts,
        g_opts,
        d_weighter=None,
        d_reducer=None,
        g_weighter=None,
        g_reducer=None,
        pre_d=None,
        post_d=None,
        pre_g=None,
        post_g=None,
        use_logits=False,
        disc_hook=None,
        gen_hook=None,
        disc_f_hook=None,
        gen_f_hook=None,
        disc_d_hook=None,
        gen_d_hook=None,
        c_hook=None,
        disc_conditions=None,
        disc_alts=None,
        gen_conditions=None,
        gen_alts=None,
        disc_domains=None,
        gen_domains=None,
        disc_domain_loss_fn=None,
        gen_domain_loss_fn=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        [pre_d, post_d, pre_g, post_g] = c_f.many_default(
            [pre_d, post_d, pre_g, post_g], [[], [], [], []]
        )
        disc_f_hook = c_f.default(
            disc_f_hook,
            FeaturesForDomainLossHook,
            {"detach": True, "use_logits": use_logits, "domains": disc_domains},
        )
        gen_f_hook = c_f.default(
            gen_f_hook,
            FeaturesForDomainLossHook,
            {"use_logits": use_logits, "domains": gen_domains},
        )
        c_hook = c_f.default(c_hook, CLossHook, {})
        disc_hook = c_f.default(
            disc_hook,
            DomainLossHook,
            {
                "d_loss_fn": disc_domain_loss_fn,
                "loss_prefix": "d_",
                "detach_features": True,
                "f_hook": disc_f_hook,
                "d_hook": disc_d_hook,
                "domains": disc_domains,
            },
        )
        gen_hook = c_f.default(
            gen_hook,
            DomainLossHook,
            {
                "d_loss_fn": gen_domain_loss_fn,
                "loss_prefix": "g_",
                "reverse_labels": True,
                "f_hook": gen_f_hook,
                "d_hook": gen_d_hook,
                "domains": gen_domains,
            },
        )
        # use gen_f_hook to get undetached features first
        disc_hook = ChainHook(
            *pre_d,
            gen_f_hook,
            disc_hook,
            *post_d,
            conditions=disc_conditions,
            alts=disc_alts
        )
        gen_hook = ChainHook(
            *pre_g, gen_hook, c_hook, *post_g, conditions=gen_conditions, alts=gen_alts
        )

        disc_hook = OptimizerHook(disc_hook, d_opts, d_weighter, d_reducer)
        gen_hook = OptimizerHook(gen_hook, g_opts, g_weighter, g_reducer)
        s_hook = SummaryHook({"d_loss": disc_hook, "g_loss": gen_hook})
        self.hook = ChainHook(disc_hook, gen_hook, s_hook)

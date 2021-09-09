import torch

from ..layers import MCDLoss
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import CLossHook
from .features import FeaturesAndLogitsHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import ApplyToListHook, ChainHook, ParallelHook, RepeatHook


class MultipleCLossHook(BaseWrapperHook):
    def __init__(self, num_c=2, loss_fn=None, detach_features=False, **kwargs):
        super().__init__(**kwargs)
        self.num_c = num_c
        loss_fn = c_f.default(loss_fn, torch.nn.CrossEntropyLoss, {"reduction": "none"})
        f_hook = FeaturesAndLogitsHook(domains=["src"], detach_features=detach_features)
        c_hook = CLossHook(loss_fn=loss_fn, detach_features=detach_features)
        c_hook = ApplyToListHook(c_hook, num_c, "_logits$")
        self.hook = ChainHook(f_hook, c_hook)


class MCDLossHook(BaseWrapperHook):
    def __init__(self, detach_features=False, minimize=True, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = c_f.default(loss_fn, MCDLoss, {})
        self.hook = FeaturesAndLogitsHook(
            domains=["target"], detach_features=detach_features
        )
        self.minimize = minimize

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        [target_logits] = c_f.extract(
            [outputs, inputs], c_f.filter(self.hook.out_keys, "_logits$")
        )
        loss = self.loss_fn(*target_logits)
        if not self.minimize:
            loss = -loss
        return {"discrepancy_loss": loss}, outputs

    def _loss_keys(self):
        return ["discrepancy_loss"]


class MCDHook(BaseWrapperHook):
    """
    Implementation of
    [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1712.02560).
    """

    def __init__(
        self,
        g_opts,
        c_opts,
        discrepancy_loss_fn=None,
        x_weighter=None,
        x_reducer=None,
        y_weighter=None,
        y_reducer=None,
        z_weighter=None,
        z_reducer=None,
        pre_x=None,
        post_x=None,
        pre_y=None,
        post_y=None,
        pre_z=None,
        post_z=None,
        repeat=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        [pre_x, post_x, pre_y, post_y, pre_z, post_z] = c_f.many_default(
            [pre_x, post_x, pre_y, post_y, pre_z, post_z], [[], [], [], [], [], []]
        )
        x = ChainHook(*pre_x, MultipleCLossHook(), *post_x)
        y = ChainHook(
            *pre_y,
            MultipleCLossHook(detach_features=True),
            MCDLossHook(
                detach_features=True, minimize=False, loss_fn=discrepancy_loss_fn
            ),
            *post_y,
        )
        z = ChainHook(*pre_z, MCDLossHook(loss_fn=discrepancy_loss_fn), *post_z)

        x = OptimizerHook(x, [*c_opts, *g_opts], x_weighter, x_reducer)
        y = OptimizerHook(y, c_opts, y_weighter, y_reducer)
        z = OptimizerHook(z, g_opts, z_weighter, z_reducer)
        s_hook = SummaryHook({"x_loss": x, "y_loss": y, "z_loss": z})
        z = RepeatHook(z, repeat, keep_only_last=True)

        self.hook = ChainHook(ParallelHook(x, y, z), s_hook)

from ..layers import VATLoss
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .features import FeaturesAndLogitsHook
from .gan import GANHook
from .losses import TargetEntropyHook
from .utils import ChainHook


class VATHook(BaseWrapperHook):
    """
    Applies the [```VATLoss```][pytorch_adapt.layers.vat_loss.VATLoss].
    """

    def __init__(self, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = c_f.default(loss_fn, VATLoss, {})
        self.hook = FeaturesAndLogitsHook()

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        [src_imgs, target_imgs, combined_model] = c_f.extract(
            inputs, ["src_imgs", "target_imgs", "combined_model"]
        )
        [src_logits, target_logits] = c_f.extract(
            [outputs, inputs],
            c_f.filter(self.hook.out_keys, "_logits$", ["^src", "^target"]),
        )
        src_vat_loss = self.loss_fn(src_imgs, src_logits, combined_model)
        target_vat_loss = self.loss_fn(target_imgs, target_logits, combined_model)
        return {
            "src_vat_loss": src_vat_loss,
            "target_vat_loss": target_vat_loss,
        }, outputs

    def _loss_keys(self):
        return ["src_vat_loss", "target_vat_loss"]


class VATPlusEntropyHook(BaseWrapperHook):
    def __init__(self, vat_loss_fn=None, entropy_loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        hook1 = VATHook(vat_loss_fn)
        hook2 = TargetEntropyHook(entropy_loss_fn)
        self.hook = ChainHook(hook1, hook2)


class VADAHook(GANHook):
    """
    Implementation of VADA from
    [A DIRT-T Approach to Unsupervised Domain Adaptation](https://arxiv.org/abs/1802.08735).
    """

    def __init__(self, vat_loss_fn=None, entropy_loss_fn=None, post_g=None, **kwargs):
        post_g = c_f.default(post_g, [])
        post_g += [VATPlusEntropyHook(vat_loss_fn, entropy_loss_fn)]
        super().__init__(post_g=post_g, **kwargs)

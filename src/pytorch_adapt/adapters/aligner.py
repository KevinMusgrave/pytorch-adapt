from ..containers import KeyEnforcer
from ..hooks import AlignerPlusCHook, RTNHook
from ..utils.common_functions import check_domain
from .base_adapter import BaseGCAdapter


class Aligner(BaseGCAdapter):
    hook_cls = AlignerPlusCHook

    def init_hook(self, hook_kwargs):
        self.hook = self.hook_cls(opts=list(self.optimizers.values()), **hook_kwargs)


class RTN(Aligner):
    hook_cls = RTNHook

    def inference_default(self, x, domain=None):
        domain = check_domain(self, domain)
        features, logits = super().inference_default(x, domain)
        if domain == 0:
            return features, self.models["residual_model"](logits)
        return features, logits

    def get_key_enforcer(self):
        ke = super().get_key_enforcer()
        ke.requirements["models"].append("residual_model")
        ke.requirements["optimizers"].append("residual_model")
        ke.requirements["misc"] = ["feature_combiner"]
        return ke

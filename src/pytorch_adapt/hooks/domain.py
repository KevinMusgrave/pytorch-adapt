import torch

from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .features import (
    DLogitsHook,
    FeaturesAndLogitsHook,
    FeaturesChainHook,
    FeaturesHook,
    LogitsHook,
)
from .utils import ChainHook


class FeaturesForDomainLossHook(FeaturesChainHook):
    def __init__(
        self,
        f_hook=None,
        l_hook=None,
        use_logits=False,
        domains=None,
        detach=False,
        **kwargs,
    ):
        hooks = [
            c_f.default(
                f_hook,
                FeaturesHook(detach=detach, domains=domains),
            )
        ]
        if use_logits:
            hooks.append(
                c_f.default(l_hook, LogitsHook(detach=detach, domains=domains))
            )
        super().__init__(*hooks, **kwargs)


class DomainLossHook(BaseWrapperHook):
    def __init__(
        self,
        d_loss_fn=None,
        detach_features=False,
        reverse_labels=False,
        domains=None,
        f_hook=None,
        d_hook=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_loss_fn = c_f.default(
            d_loss_fn, torch.nn.BCEWithLogitsLoss, {"reduction": "none"}
        )
        self.reverse_labels = reverse_labels
        self.domains = c_f.default(domains, ["src", "target"])
        f_hook = c_f.default(
            f_hook,
            FeaturesForDomainLossHook,
            {"detach": detach_features, "domains": domains},
        )
        d_hook = c_f.default(d_hook, DLogitsHook, {"domains": domains})
        f_out = f_hook.last_hook_out_keys
        d_in = d_hook.in_keys
        d_hook.set_in_keys(f_out)
        self.check_fhook_dhook_keys(f_hook, d_hook, detach_features)
        self.hook = ChainHook(f_hook, d_hook)
        self.in_keys = self.hook.in_keys + ["src_domain", "target_domain"]

    def call(self, losses, inputs):
        losses = {}
        outputs = self.hook(losses, inputs)[1]
        labels = self.extract_domain_labels(inputs)
        for domain_name, labels in labels.items():
            [dlogits] = c_f.extract(
                [outputs, inputs],
                c_f.filter(self.hook.out_keys, f"_dlogits$", [f"^{domain_name}"]),
            )
            if dlogits.dim() > 1:
                labels = labels.type(torch.long)
            else:
                labels = labels.type(torch.float)
            loss = self.d_loss_fn(dlogits, labels)
            losses[f"{domain_name}_domain_loss"] = loss
        return losses, outputs

    def extract_domain_labels(self, inputs):
        [src_domain, target_domain] = c_f.extract(
            inputs, ["src_domain", "target_domain"]
        )
        if self.reverse_labels:
            labels = {"src": target_domain, "target": src_domain}
        else:
            labels = {"src": src_domain, "target": target_domain}
        return {k: v for k, v in labels.items() if k in self.domains}

    def _loss_keys(self):
        return [f"{x}_domain_loss" for x in self.domains]

    def check_fhook_dhook_keys(self, f_hook, d_hook, detach_features):
        if detach_features and len(
            c_f.filter(f_hook.out_keys, "detached$", self.domains)
        ) < len(self.domains):
            error_str = (
                "detach_features is True, but the number of f_hook's detached outputs "
            )
            error_str += "doesn't match the number of domains."
            error_str += f"\nf_hook's outputs: {f_hook.out_keys}"
            error_str += f"\nfdomains: {self.domains}"
            raise ValueError(error_str)
        for name, keys in [("f_hook", f_hook.out_keys), ("d_hook", d_hook.out_keys)]:
            if not all(
                c_f.filter(keys, f"^{self.domains[i]}")
                for i in range(len(self.domains))
            ):
                raise ValueError(
                    f"domains = {self.domains} but d_hook.out_keys = {d_hook.out_keys}"
                )

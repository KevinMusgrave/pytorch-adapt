from contextlib import nullcontext

import torch

from pytorch_adapt.hooks import AFNHook, BNMHook, BSPHook, MCCHook
from pytorch_adapt.layers import (
    AdaptiveFeatureNorm,
    BatchSpectralLoss,
    BNMLoss,
    EntropyWeights,
    MaxNormalizer,
    MCCLoss,
)


class Net(torch.nn.Module):
    def __init__(self, in_size, out_size, with_batch_norm=False):
        super().__init__()
        self.count = 0
        if with_batch_norm:
            self.net = torch.nn.Sequential(
                torch.nn.BatchNorm1d(in_size), torch.nn.Linear(in_size, out_size)
            )
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(in_size, out_size))

    def forward(self, x):
        self.count += 1
        x = self.net(x)
        if x.dim() > 1 and x.shape[1] == 1:
            x = x.squeeze(1)
        return x


def check_requires_grad(v, rg):
    if isinstance(v, list):
        return all(v2.requires_grad == rg for v2 in v if v is not None)
    else:
        return v.requires_grad == rg


def assertRequiresGrad(cls, outputs):
    cls.assertTrue(
        all(
            check_requires_grad(v, False)
            for k, v in outputs.items()
            if k.endswith("detached") and v is not None
        )
    )
    cls.assertTrue(
        all(
            check_requires_grad(v, True)
            for k, v in outputs.items()
            if not k.endswith("detached") and v is not None
        )
    )


def get_models_and_data(with_batch_norm=False, d_out=1, d_uses_logits=False):
    src_domain = torch.randint(0, 2, size=(100,)).float()
    target_domain = torch.randint(0, 2, size=(100,)).float()
    src_labels = torch.randint(0, 10, size=(100,))
    src_imgs = torch.randn(100, 32)
    target_imgs = torch.randn(100, 32)
    G = Net(32, 16, with_batch_norm)
    C = Net(16, 10, with_batch_norm)
    if d_uses_logits:
        D = Net(10, d_out, with_batch_norm)
    else:
        D = Net(16, d_out, with_batch_norm)
    return G, C, D, src_imgs, src_labels, target_imgs, src_domain, target_domain


def get_opts(models):
    return {
        f"{k}_opt": torch.optim.SGD(v.parameters(), lr=0.1) for k, v in models.items()
    }


def post_g_hook_update_keys(post_g, loss_keys, output_keys):
    if isinstance(post_g, BSPHook):
        loss_keys.update({"bsp_loss"})
        for domain in post_g.domains:
            output_keys.update({f"{domain}_imgs_features"})
    elif isinstance(post_g, BNMHook):
        output_keys.update({"target_imgs_features_logits"})
        loss_keys.update({"bnm_loss"})
    elif isinstance(post_g, MCCHook):
        output_keys.update({"target_imgs_features_logits"})
        loss_keys.update({"mcc_loss"})
    elif isinstance(post_g, AFNHook):
        output_keys.update({"src_imgs_features", "target_imgs_features"})
        loss_keys.update({"afn_loss"})


def post_g_hook_update_total_loss(
    post_g, total_loss, src_features, target_features, target_logits
):
    if isinstance(post_g, BSPHook):
        bsp_loss = 0
        if src_features is not None:
            bsp_loss += BatchSpectralLoss()(src_features)
        if target_features is not None:
            bsp_loss += BatchSpectralLoss()(target_features)
        total_loss.append(bsp_loss)
    elif isinstance(post_g, BNMHook):
        if target_logits is not None:
            total_loss.append(BNMLoss()(target_logits))
    elif isinstance(post_g, MCCHook):
        if target_logits is not None:
            total_loss.append(MCCLoss()(target_logits))
    elif isinstance(post_g, AFNHook):
        loss_fn = AdaptiveFeatureNorm()
        total_loss.append(loss_fn(src_features) + loss_fn(target_features))


def get_entropy_weights(c_logits, bs, detach_reducer):
    entropy_weighter = EntropyWeights(normalizer=MaxNormalizer(detach=True))
    entropy_context = torch.no_grad() if detach_reducer else nullcontext()

    with entropy_context:
        src_entropy_weights = entropy_weighter(c_logits[:bs])
        target_entropy_weights = entropy_weighter(c_logits[bs:])

    if c_logits.requires_grad:
        if any(
            x.requires_grad != (not detach_reducer)
            for x in [src_entropy_weights, target_entropy_weights]
        ):
            raise ValueError
    elif any(x.requires_grad for x in [src_entropy_weights, target_entropy_weights]):
        raise ValueError

    return src_entropy_weights, target_entropy_weights

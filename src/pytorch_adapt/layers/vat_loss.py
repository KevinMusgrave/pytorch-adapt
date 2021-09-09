import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f

# references:
# https://github.com/takerum/vat_chainer/blob/master/source/chainer_functions/loss.py
# https://github.com/takerum/vat_tf/blob/master/vat.py
# https://github.com/RuiShu/dirt-t/blob/master/codebase/models/extra_layers.py
# https://github.com/lyakaap/VAT-pytorch
# https://github.com/9310gaurav/virtual-adversarial-training


def get_normalized_noise(noise):
    original_shape = noise.shape
    noise = F.normalize(noise.view(original_shape[0], -1), dim=1)
    return noise.view(*original_shape)


class VATLoss(torch.nn.Module):
    """
    Implementation of the loss used in

    - [Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning](https://arxiv.org/abs/1704.03976)
    - [A DIRT-T Approach to Unsupervised Domain Adaptation](https://arxiv.org/abs/1802.08735)
    """

    def __init__(
        self, num_power_iterations: int = 1, xi: float = 1e-6, epsilon: float = 8.0
    ):
        """
        Arguments:
            num_power_iterations: The number of iterations for
                computing the approximation of the adversarial perturbation.
            xi: The L2 norm of the the generated noise which is used
                in the process of creating the perturbation.
            epsilon: The L2 norm of the generated perturbation.
        """
        super().__init__()
        self.num_power_iterations = num_power_iterations
        self.xi = xi
        self.epsilon = epsilon
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        pml_cf.add_to_recordable_attributes(
            self, list_of_names=["num_power_iterations", "xi", "epsilon"]
        )

    def forward(
        self, imgs: torch.Tensor, logits: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Arguments:
            imgs: The input to the model
            logits: The model's logits computed from ```imgs```
            model: The aforementioned model
        """
        logits = logits.detach()
        model.apply(c_f.set_layers_mode("eval", c_f.batchnorm_types()))
        perturbation = self.get_perturbation(imgs, logits, model)
        new_logits = model(imgs + perturbation)

        preds = F.softmax(logits, dim=1)
        new_preds = F.log_softmax(new_logits, dim=1)
        model.apply(c_f.set_layers_mode("train", c_f.batchnorm_types()))
        return self.kl_div(new_preds, preds)

    def get_perturbation(self, imgs, original_logits, model):
        noise = torch.randn(*imgs.shape, device=original_logits.device)
        original_preds = F.softmax(original_logits, dim=1)

        for _ in range(self.num_power_iterations):
            noise.requires_grad = True
            noise = self.xi * get_normalized_noise(noise)
            noise.retain_grad()
            new_preds = F.log_softmax(model(imgs + noise), dim=1)
            dist = self.kl_div(new_preds, original_preds)
            dist.backward(retain_graph=True)
            noise = noise.grad.detach()
            model.zero_grad()

        return self.epsilon * get_normalized_noise(noise)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["num_power_iterations", "xi", "epsilon"])

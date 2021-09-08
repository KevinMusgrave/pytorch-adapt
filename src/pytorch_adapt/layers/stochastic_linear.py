import torch

from ..utils import common_functions as c_f


# https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Stochastic_Classifiers_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf
class StochasticLinear(torch.nn.Module):
    """
    Implementation of the stochastic layer from
    [Stochastic Classifiers for Unsupervised Domain Adaptation](https://xiatian-zhu.github.io/papers/LuEtAl_CVPR2020.pdf).
    In ```train()``` mode, it uses random weights and biases
    that are sampled from a learned normal distribution.
    In ```eval()``` mode, the learned mean is used.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Arguments:
            in_features: size of each input sample
            out_features: size of each output sample
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = torch.nn.Parameter(
            torch.rand(in_features, out_features, **factory_kwargs)
        )
        self.weight_sigma = torch.nn.Parameter(
            torch.rand(in_features, out_features, **factory_kwargs)
        )
        self.bias_mean = torch.nn.Parameter(torch.rand(out_features, **factory_kwargs))
        self.bias_sigma = torch.nn.Parameter(torch.rand(out_features, **factory_kwargs))

    def random_sample(self, mean, sigma):
        eps = torch.randn(*sigma.shape, device=sigma.device, dtype=sigma.dtype)
        return mean + (sigma * eps)

    def forward(self, x):
        """"""
        if self.training:
            weight = self.random_sample(self.weight_mean, self.weight_sigma)
            bias = self.random_sample(self.bias_mean, self.bias_sigma)
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return torch.matmul(x, weight) + bias

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["in_features", "out_features"])

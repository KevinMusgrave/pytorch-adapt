import unittest

import torch

from pytorch_adapt.layers import StochasticLinear

from .. import TEST_DEVICE, TEST_DTYPES


def check(cls, layer, x, batch_size, out_size, should_be_equal):
    prev_y = None
    for _ in range(10):
        y = layer(x)
        cls.assertTrue(y.shape == torch.Size([batch_size, out_size]))
        if prev_y is not None:
            e = torch.equal(y, prev_y)
            if should_be_equal:
                cls.assertTrue(e)
            else:
                cls.assertTrue(not e)
        prev_y = y


class TestStochasticLinear(unittest.TestCase):
    def test_stochastic_linear(self):
        torch.manual_seed(256)
        for dtype in TEST_DTYPES:
            batch_size = 32
            in_size = 128
            out_size = 10
            layer = StochasticLinear(in_size, out_size, device=TEST_DEVICE, dtype=dtype)

            x = torch.randn(
                batch_size, in_size, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)

            check(self, layer, x, batch_size, out_size, False)
            layer.eval()
            check(self, layer, x, batch_size, out_size, True)

            equivalent_layer = torch.nn.Linear(in_size, out_size)
            equivalent_layer.weight = torch.nn.Parameter(layer.weight_mean.t())
            equivalent_layer.bias = torch.nn.Parameter(layer.bias_mean.t())
            equivalent_layer.eval()

            y = layer(x)
            z = equivalent_layer(x)
            self.assertTrue(torch.allclose(y, z, rtol=1e-2))

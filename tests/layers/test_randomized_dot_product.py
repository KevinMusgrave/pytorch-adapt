import unittest

import torch

from pytorch_adapt.layers import RandomizedDotProduct

from .. import TEST_DEVICE, TEST_DTYPES


class TestRandomizedDotProduct(unittest.TestCase):
    def test_randomized_dot_product(self):
        for dtype in TEST_DTYPES:
            in_dims = [512, 128]
            out_dim = 1234
            fn = RandomizedDotProduct(in_dims, out_dim)
            batch_size = 32
            x = torch.randn(batch_size, in_dims[0], device=TEST_DEVICE).type(dtype)
            y = torch.randn(batch_size, in_dims[1], device=TEST_DEVICE).type(dtype)
            combined = fn(x, y)

            correct = torch.mm(x, fn.rand_mat0) * torch.mm(y, fn.rand_mat1)
            correct /= torch.sqrt(torch.tensor(out_dim, device=TEST_DEVICE).type(dtype))

            self.assertTrue(torch.all(torch.isclose(combined, correct, rtol=1e-2)))
            self.assertTrue(combined.shape == torch.Size([batch_size, out_dim]))

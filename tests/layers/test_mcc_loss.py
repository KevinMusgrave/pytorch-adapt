import unittest

import torch
import torch.nn as nn
from pytorch_metric_learning.utils import common_functions as pml_cf

from pytorch_adapt.layers import MCCLoss

from .. import TEST_DEVICE, TEST_DTYPES


# https://github.com/thuml/Versatile-Domain-Adaptation
def Entropy(input_):
    epsilon = pml_cf.small_val(input_.dtype)
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


# https://github.com/thuml/Versatile-Domain-Adaptation/blob/master/pytorch/train_image_office.py
def original_implementation(x, temperature, class_num, batch_size):
    outputs_target_temp = x / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = (
        batch_size * target_entropy_weight / torch.sum(target_entropy_weight)
    )
    cov_matrix_t = (
        target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1))
        .transpose(1, 0)
        .mm(target_softmax_out_temp)
    )
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    return (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num


class TestMCCLoss(unittest.TestCase):
    def test_mcc_loss(self):
        batch_size = 32
        class_num = 100
        for T in [0.5, 1, 2.5]:
            for dtype in TEST_DTYPES:
                loss_fn = MCCLoss(T=T)
                x = torch.randn(
                    batch_size, class_num, device=TEST_DEVICE, requires_grad=True
                ).type(dtype)
                x.retain_grad()
                loss = loss_fn(x)
                correct_loss = original_implementation(x, T, class_num, batch_size)
                self.assertTrue(torch.isclose(loss, correct_loss))

                loss.backward()
                grad1 = x.grad.clone()
                x.grad = None

                correct_loss.backward()
                grad2 = x.grad.clone()

                self.assertTrue(torch.allclose(grad1, grad2, rtol=1e-2))

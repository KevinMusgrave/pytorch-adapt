import unittest

import torch

from pytorch_adapt.layers import GradientReversal

from .. import TEST_DEVICE, TEST_DTYPES


class TestGradientReversal(unittest.TestCase):
    def helper(self, model, input_data, loss_fn, truth, optimizer, step_optimizer):
        output = model(input_data)
        loss = loss_fn(output, truth)
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        if step_optimizer:
            optimizer.step()
        optimizer.zero_grad()
        return gradients

    def test_gradient_reversal(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        input_data = torch.randn(64, 500, device=TEST_DEVICE)
        truth = torch.randint(low=0, high=100, size=(64,), device=TEST_DEVICE)
        lr = 1

        for reversal_weight in [0.1, 1, 10]:
            model = torch.nn.Linear(500, 100).to(TEST_DEVICE)
            reversal_model = torch.nn.Sequential(
                model, GradientReversal(reversal_weight)
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            reversal_optimizer = torch.optim.SGD(reversal_model.parameters(), lr=lr)

            for step_optimizer in [False, True]:
                for dtype in TEST_DTYPES:
                    input_data = input_data.type(dtype)
                    model = model.type(dtype)
                    gradients = self.helper(
                        model, input_data, loss_fn, truth, optimizer, step_optimizer
                    )
                    original_weight, original_bias = (
                        model.weight.clone(),
                        model.bias.clone(),
                    )
                    reversed_gradients = self.helper(
                        reversal_model,
                        input_data,
                        loss_fn,
                        truth,
                        reversal_optimizer,
                        step_optimizer,
                    )

                    if step_optimizer:
                        original_weight_shifted = (
                            original_weight.type(dtype) - reversed_gradients[0] * lr
                        )
                        original_bias_shifted = (
                            original_bias.type(dtype) - reversed_gradients[1] * lr
                        )
                        self.assertTrue(
                            torch.all(
                                torch.isclose(model.weight, original_weight_shifted)
                            )
                        )
                        self.assertTrue(
                            torch.all(torch.isclose(model.bias, original_bias_shifted))
                        )
                    else:
                        if dtype == torch.float16:
                            continue
                        for x, y in zip(gradients, reversed_gradients):
                            self.assertTrue(
                                torch.all(
                                    torch.isclose(x * reversal_weight, -y, rtol=1e-2)
                                )
                            )

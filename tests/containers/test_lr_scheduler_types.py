import unittest

import torch
import torchvision

from pytorch_adapt.containers import LRSchedulers, Models, Optimizers


class TestLRSchedulerTypes(unittest.TestCase):
    def test_lr_scheduler_types(self):
        lr = 0.1
        models = Models(
            {
                "A": torchvision.models.resnet18(),
                "B": torchvision.models.alexnet(),
                "C": torchvision.models.resnet34(),
            }
        )

        optimizers = Optimizers((torch.optim.Adam, {"lr": lr}))
        optimizers.create_with(models)

        gamma = 0.5
        # test lr scheduler
        lr_schedulers = LRSchedulers(
            {
                "A": torch.optim.lr_scheduler.StepLR(
                    optimizers["A"], step_size=1, gamma=gamma
                ),
                "B": torch.optim.lr_scheduler.StepLR(
                    optimizers["B"], step_size=1, gamma=gamma
                ),
            },
            scheduler_types={"per_step": ["A"], "per_epoch": ["B"]},
        )

        lr_schedulers2 = LRSchedulers(
            {
                "C": torch.optim.lr_scheduler.StepLR(
                    optimizers["C"], step_size=1, gamma=gamma
                ),
            },
            scheduler_types={"per_step": ["C"]},
        )

        lr_schedulers.merge(lr_schedulers2)

        num_steps = 5
        for _ in range(num_steps):
            optimizers.step()
            lr_schedulers.step("per_step")

        self.assertTrue(
            all(
                optimizers[k].param_groups[0]["lr"] == lr * (0.5 ** num_steps)
                for k in ["A", "C"]
            )
        )

        self.assertTrue(optimizers["B"].param_groups[0]["lr"] == lr)

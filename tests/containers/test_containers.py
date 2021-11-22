import unittest

import torch
import torchvision

from pytorch_adapt.containers import (
    LRSchedulers,
    Misc,
    Models,
    MultipleContainers,
    Optimizers,
)
from pytorch_adapt.hooks import EmptyHook
from pytorch_adapt.layers import DoNothingOptimizer
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.weighters import MeanWeighter, SumWeighter


class TestContainers(unittest.TestCase):
    def test_containers(self):
        models = Models(
            {
                "A": torch.nn.Identity(),
                "B": (torchvision.models.resnet18, {}),
                "C": torchvision.models.alexnet(),
            }
        )
        models.create()

        optimizer_tuple = (torch.optim.Adam, {"lr": 0.1})

        oc = Optimizers(optimizer_tuple, multipliers={"D": 10})
        self.assertRaises(KeyError, lambda: oc.create_with(models))
        oc = Optimizers(optimizer_tuple, multipliers={"C": 10})
        oc.create_with(models)
        self.assertTrue(c_f.get_lr(oc["C"]) == 1)
        self.assertTrue(c_f.get_lr(oc["B"]) == 0.1)

        # tuple input

        for M in [None, models]:
            oc = Optimizers(optimizer_tuple, M)
            if not M:
                oc.create_with(models)
            self.assertTrue(isinstance(oc["A"], DoNothingOptimizer))
            self.assertTrue(isinstance(oc["B"], torch.optim.Adam))
            self.assertTrue(isinstance(oc["C"], torch.optim.Adam))

        # dictionary input
        spec1 = {"B": torch.optim.SGD(models["B"].parameters(), lr=0.1)}
        spec2 = {"B": (torch.optim.SGD, {"lr": 0.1})}
        for M in [None, models]:
            for spec in [spec1, spec2]:
                oc = Optimizers(spec, M)
                if not M:
                    oc.create_with(models)
                self.assertTrue(isinstance(oc["B"], torch.optim.SGD))

        # test merging
        oc2 = Optimizers(optimizer_tuple, models)
        oc2.merge(oc)
        self.assertTrue(list(oc2.keys()) == ["A", "B", "C"])
        self.assertTrue(isinstance(oc2["A"], DoNothingOptimizer))
        self.assertTrue(isinstance(oc2["B"], torch.optim.SGD))
        self.assertTrue(isinstance(oc2["C"], torch.optim.Adam))

        # test lr scheduler
        lr = LRSchedulers(
            {
                "B": (torch.optim.lr_scheduler.StepLR, {"step_size": 2}),
                "C": torch.optim.lr_scheduler.MultiStepLR(oc2["C"], [0, 1, 2]),
            }
        )
        lr.create_with(oc2)
        self.assertTrue(list(lr.keys()) == ["B", "C"])
        self.assertTrue(isinstance(lr["B"], torch.optim.lr_scheduler.StepLR))
        self.assertTrue(isinstance(lr["C"], torch.optim.lr_scheduler.MultiStepLR))

        # test merging dict with tuple
        models1 = Models(
            {
                "A": torch.nn.Identity(),
            }
        )
        models2 = Models((torchvision.models.resnet18, {}))
        models1.merge(models2)
        self.assertTrue(list(models1.keys()) == ["A"])
        self.assertTrue(models1["A"] == (torchvision.models.resnet18, {}))

        # another test merging dict with tuple
        models1 = Models(
            {
                "A": torch.nn.Identity(),
            }
        )
        models2 = Models((torchvision.models.resnet18, {}), keys=["B", "C"])
        models1.merge(models2)
        self.assertTrue(list(models1.keys()) == ["A", "B", "C"])
        self.assertTrue(models1.store_as_tuple is None)

        # merge container that has keys with container that has a tuple
        oc1 = Optimizers(
            (torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
            keys=["feature_extractor", "classifier", "discriminator"],
        )
        oc2 = Optimizers((torch.optim.Adam, {"lr": 0.01, "momentum": 0.9}))
        oc1.merge(oc2)
        self.assertTrue(all(x[0] == torch.optim.Adam for x in oc1.values()))

        # merge container that has keys with container that has a tuple
        oc1 = Optimizers(
            (torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
            keys=["feature_extractor", "classifier", "discriminator"],
        )
        oc2 = Optimizers(
            (torch.optim.Adam, {"lr": 0.01, "momentum": 0.9}), keys=["some_other_key"]
        )
        oc1.merge(oc2)
        self.assertTrue(
            all(
                oc1[x][0] == torch.optim.SGD
                for x in ["feature_extractor", "classifier", "discriminator"]
            )
        )
        self.assertTrue(all(oc1[x][0] == torch.optim.Adam for x in ["some_other_key"]))

        # merge container that has keys with container that has a tuple
        w1 = Misc((MeanWeighter, {}), keys=["A", "B"])
        w2 = Misc({"A": SumWeighter()})
        w1.merge(w2)
        w1.create()
        self.assertTrue(isinstance(w1["A"], SumWeighter))
        self.assertTrue(isinstance(w1["B"], MeanWeighter))

        # test function, no kwargs, and keys
        hooks = Misc((EmptyHook, {}), keys=["A", "B"])
        hooks.create()
        self.assertTrue(all(isinstance(hooks[x], EmptyHook) for x in ["A", "B"]))

        with self.assertRaises(ValueError):
            hooks = Misc((EmptyHook, 1, 2), keys=["A", "B"])
            hooks.create()

    def test_merge_with_non_container(self):
        x = MultipleContainers()
        for non_container in [[], {}]:
            with self.assertRaises(TypeError):
                x.merge(optimizers=non_container)

        default_opt = (torch.optim.Adam, {"lr": 0.1})
        x = MultipleContainers(optimizers=Optimizers(default_opt))
        x.merge(optimizers=None)
        x["optimizers"].store == default_opt

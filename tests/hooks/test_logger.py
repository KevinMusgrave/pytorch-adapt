import unittest

import torch

from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.hooks.logger import HookLogger


class TestHookLogger(unittest.TestCase):
    def test_basic(self):
        test_str = "test test"
        correct_out = f"{__name__}: {test_str}"
        logger = HookLogger(__name__)
        logger(test_str)
        self.assertTrue(logger.str == correct_out)
        logger.reset()
        self.assertTrue(logger.str == "")
        logger(test_str)
        self.assertTrue(logger.str == correct_out)

    def test_bsp_hook(self):
        hook = DANNHook(opts=[])
        data = {
            "src_imgs": torch.randn(32, 32),
            # "target_imgs": torch.randn(32, 32),
            "src_domain": torch.zeros(32),
            "target_domain": torch.ones(32),
            "src_labels": torch.randint(0, 2, size=(32,)),
        }
        models = {
            "G": torch.nn.Linear(32, 10),
            "C": torch.nn.Linear(10, 2),
            "D": torch.nn.Sequential(torch.nn.Linear(10, 1), torch.nn.Flatten(0)),
        }
        # with self.assertRaises(KeyError):
        hook({}, {**models, **data})
        # print(hook.logger.str)

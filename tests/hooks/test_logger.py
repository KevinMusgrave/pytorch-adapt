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
        }
        models = {
            "G": torch.nn.Linear(32, 10),
            "C": torch.nn.Linear(10, 2),
            "D": torch.nn.Sequential(torch.nn.Linear(10, 1), torch.nn.Flatten(0)),
        }
        with self.assertRaises(KeyError) as cm:
            hook({}, {**data, **models})

        correct_str = (
            "in DANNHook: __call__"
            "\nin ChainHook: __call__"
            "\nin OptimizerHook: __call__"
            "\nin ChainHook: __call__"
            "\nin FeaturesForDomainLossHook: __call__"
            "\nin FeaturesHook: __call__"
            "\nFeaturesHook: Getting src"
            "\nFeaturesHook: Getting output ['src_imgs_features']"
            "\nFeaturesHook: Using model G with inputs ['src_imgs']"
            "\nFeaturesHook: Getting target"
            "\nFeaturesHook: Getting output ['target_imgs_features']"
            "\nFeaturesHook: Using model G with inputs ['target_imgs']"
            "\ntarget_imgs"
        )

        self.assertTrue(str(cm.exception) == correct_str)

import unittest

from pytorch_adapt.hooks.logger import HOOK_STREAM, HookLogger, reset_hook_logger


class TestHookLogger(unittest.TestCase):
    def test_hook_logger(self):
        test_str = "test test"
        correct_out = f"{__name__}: {test_str}\n"
        logger = HookLogger(__name__)
        logger(test_str)
        self.assertTrue(HOOK_STREAM.getvalue() == correct_out)
        reset_hook_logger()
        self.assertTrue(HOOK_STREAM.getvalue() == "")
        logger(test_str)
        self.assertTrue(HOOK_STREAM.getvalue() == correct_out)

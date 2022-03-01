import unittest

from pytorch_adapt.hooks import (
    BSPHook,
    ClassifierHook,
    CLossHook,
    FeaturesAndLogitsHook,
    FeaturesHook,
    get_hooks,
)


class TestGetHooks(unittest.TestCase):
    def test_simple(self):
        hooks = get_hooks(BSPHook(), FeaturesHook)
        self.assertTrue(len(hooks) == 1)

        hooks = get_hooks(BSPHook(), FeaturesAndLogitsHook)
        self.assertTrue(len(hooks) == 0)

    def test_deeply_nested(self):
        hook = ClassifierHook(opts=[], post=[BSPHook()])
        hooks = get_hooks(hook, BSPHook)
        self.assertTrue(len(hooks) == 1)

        hooks = get_hooks(hook, CLossHook)
        self.assertTrue(len(hooks) == 1)

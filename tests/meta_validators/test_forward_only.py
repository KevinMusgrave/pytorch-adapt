import unittest

from pytorch_adapt.meta_validators import ForwardOnlyValidator
from pytorch_adapt.validators import AccuracyValidator, WithHistory

from ..adapters.get_dann import get_dann


class TestForwardOnlyValidator(unittest.TestCase):
    def test_forward_only_validator(self):
        mv = ForwardOnlyValidator()
        validator = WithHistory(AccuracyValidator())
        adapter, datasets = get_dann(validator=validator)
        output = mv.run(
            adapter=adapter,
            datasets=datasets,
        )
        self.assertTrue(output[0] == validator.best_score)
        self.assertTrue(output[1] == validator.best_epoch)

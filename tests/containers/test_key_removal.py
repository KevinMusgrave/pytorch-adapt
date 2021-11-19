import unittest

import torch

from pytorch_adapt.containers import DeleteKey, Models, Optimizers

from ..adapters.utils import Discriminator, get_source_model


class TestKeyRemoval(unittest.TestCase):
    def test_key_removal(self):

        num_classes = 10
        source_model, source_classifier = get_source_model()
        discriminator = Discriminator(source_classifier.net[0].in_features, 1)
        models = Models(
            {
                "feature_extractor": source_model,
                "classifier": source_classifier,
                "discriminator": discriminator,
            }
        )
        optimizers1 = Optimizers((torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}))
        optimizers1.create_with(models)
        self.assertTrue(
            optimizers1.keys() == {"feature_extractor", "classifier", "discriminator"}
        )

        optimizers1 = Optimizers((torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}))
        optimizers2 = Optimizers({"feature_extractor": DeleteKey()})
        optimizers1.merge(optimizers2)
        optimizers1.create_with(models)
        self.assertTrue(optimizers1.keys() == {"classifier", "discriminator"})

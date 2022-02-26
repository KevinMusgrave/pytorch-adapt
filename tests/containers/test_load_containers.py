import os
import shutil
import unittest

import torch

from pytorch_adapt.containers import Models
from pytorch_adapt.containers.base_container import containers_are_equal
from pytorch_adapt.utils import common_functions as c_f

from .. import TEST_FOLDER


def get_models():
    models = {"G": torch.nn.Linear(100, 10), "C": torch.nn.Linear(10, 10)}
    return Models(models)


def are_equal(x, y):
    output = containers_are_equal(x, y)

    for k in ["G", "C"]:
        output &= c_f.state_dicts_are_equal(x[k].state_dict(), y[k].state_dict())

    return output


class TestLoadContainers(unittest.TestCase):
    def test_load_containers(self):
        save_to = os.path.join(TEST_FOLDER, "container.pt")
        models1 = get_models()
        c_f.makedir_if_not_there(TEST_FOLDER)
        torch.save(models1.state_dict(), save_to)

        models2 = get_models()
        self.assertTrue(not are_equal(models1, models2))
        models2.load_state_dict(torch.load(save_to))
        self.assertTrue(are_equal(models1, models2))

        shutil.rmtree(TEST_FOLDER)

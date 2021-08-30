import unittest

from pytorch_adapt.hooks.base import replace_mapped_keys
from pytorch_adapt.utils import common_functions as c_f


class TestBase(unittest.TestCase):
    def test_replace_mapped_keys(self):
        x_dict = {"src_combined_features": 1, "target_combined_features": 100, "G": 5}
        for key_map in [
            {
                "src_combined_features": "src_features",
                "target_combined_features": "target_features",
            },
            {"G": "C"},
        ]:
            for x in [x_dict]:
                y = c_f.map_keys(x, key_map)
                z = replace_mapped_keys(y, key_map)
                z_list = replace_mapped_keys(list(y.keys()), key_map)
                self.assertTrue(z.keys() == x.keys())
                self.assertTrue(set(z_list) == x.keys())

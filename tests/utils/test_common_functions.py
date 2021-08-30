import unittest

from pytorch_adapt.utils import common_functions as c_f


class TestCommonFunctions(unittest.TestCase):
    def test_map_keys(self):
        x = {"a": 1, "b": 2, "c": 3}
        key_map = {"c": "b"}
        new_x = c_f.map_keys(x, key_map)
        self.assertTrue(new_x == {"a": 1, "b": 3})

        x = {"a": 1, "c": 2, "b": 3}
        key_map = {"c": "b"}
        new_x = c_f.map_keys(x, key_map)
        self.assertTrue(new_x == {"a": 1, "b": 2})

        x = {"a": 1, "c": 2, "b": 3, "d": 4}
        key_map = {"c": "b", "b": "c", "d": "a", "a": "d"}
        new_x = c_f.map_keys(x, key_map)
        self.assertTrue(new_x == {"d": 1, "b": 2, "c": 3, "a": 4})

        x = {"a": 1, "c": 2, "b": 3, "d": 4}
        key_map = {}
        new_x = c_f.map_keys(x, key_map)
        self.assertTrue(new_x == x)

        key_map = {"e": 10}
        with self.assertRaises(KeyError):
            new_x = c_f.map_keys(x, key_map)

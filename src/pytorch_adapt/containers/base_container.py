from collections.abc import MutableMapping

from ..utils import common_functions as c_f


# https://stackoverflow.com/a/3387975
class BaseContainer(MutableMapping):
    """
    The parent class of all containers.
    """

    def __init__(self, store, other=None, keys=None):
        """
        Arguments:
            store: A tuple or dictionary

                - A tuple consists of ```(<class_ref>, <init kwargs>)```.
                For example, ```(torch.optim.Adam, {"lr": 0.1})```
                - A dictionary maps from object name to either a tuple
                or a fully constructed object.

            other: Another container which is used in the process
                of creating objects in this container, (e.g
                optimizers require model parameters).

            keys: Converts ```store``` from tuple to dict format,
                where each dict value is the tuple. This only works
                if ```store``` is passed in as a tuple.
        """
        if not isinstance(store, (tuple, dict)):
            raise TypeError("BaseContainer input must be a tuple or dict")
        if isinstance(store, tuple):
            self.store_as_tuple = store
            self.store = {}
        else:
            self.store_as_tuple = None
            self.store = store
        if keys is not None:
            self.duplicate(keys)
        if other is not None:
            self.create_with(other)

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key

    def __repr__(self):
        if isinstance(self.store, dict):
            output = ""
            for k, v in self.items():
                output += f"{k}: {v}\n"
            return output
        return str(self.store)

    def merge(self, other: "BaseContainer"):
        """
        Merges another container into this one.
        Arguments:
            other: The container that will be merged into this container.
        """
        if not isinstance(other, BaseContainer):
            raise TypeError("merge can only be done with another container")
        if other.store_as_tuple:
            if len(self) > 0:
                for k, v in self.items():
                    self[k] = other.store_as_tuple
            else:
                self.store_as_tuple = other.store_as_tuple
        else:
            for k, v in other.items():
                self[k] = v

    def create(self):
        """
        Initializes objects by converting all
        tuples in the store into objects.
        """
        for k, v in self.items():
            if isinstance(v, tuple):
                if len(v) == 2:
                    class_ref, kwargs = v
                    self[k] = class_ref(**kwargs)
                elif len(v) == 1:
                    self[k] = v[0]
                else:
                    raise ValueError(
                        f"The tuple {v} has length={len(v)}, but it must be of length 1 or 2"
                    )
        self.delete_unwanted_keys()

    def create_with(self, other):
        """
        Initializes objects conditioned on the input container.
        """
        self.store_as_tuple = self.type_check(self.store_as_tuple, other)
        self.store = self.type_check(self.store, other)
        self.store_as_tuple.update(self.store)
        self.store = self.store_as_tuple
        self.store_as_tuple = None
        self.delete_unwanted_keys()
        self._create_with(other)

    def _create_with(self, other):
        pass

    def type_check(self, store, other):
        if isinstance(store, tuple):
            return {k: store for k in other.keys()}
        elif isinstance(store, dict):
            return store
        elif store is None:
            return {}

    def duplicate(self, keys):
        if isinstance(self.store_as_tuple, tuple):
            self.store = {k: self.store_as_tuple for k in keys}
            self.store_as_tuple = None
        else:
            raise TypeError("If keys are specified, store must be a tuple.")

    def apply(self, function, keys=None):
        if keys is None:
            keys = list(self.keys())
        for k in keys:
            self[k] = function(self[k])

    def delete_unwanted_keys(self):
        del_list = []
        for k, v in self.items():
            if isinstance(v, DeleteKey) or (isinstance(v, tuple) and v[0] == DeleteKey):
                del_list.append(k)
        for k in del_list:
            del self[k]


class DeleteKey:
    pass


def containers_are_equal(x, y):
    if not isinstance(x, BaseContainer) or not isinstance(y, BaseContainer):
        raise TypeError("inputs must be Container types")
    if x.keys() != y.keys():
        return False
    for k, v1 in x.items():
        v2 = y[k]
        if type(v1) != type(v2):
            return False
        elif isinstance(v1, BaseContainer):
            if not containers_are_equal(v1, v2):
                return False
        elif not c_f.has_state_dict(v1):
            if v1.__dict__ != v2.__dict__:
                return False
        elif c_f.has_state_dict(v1):
            if not c_f.state_dicts_are_equal(v1.state_dict(), v2.state_dict()):
                return False
    return True

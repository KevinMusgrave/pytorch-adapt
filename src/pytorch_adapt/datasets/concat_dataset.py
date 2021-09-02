import torch

from ..utils import common_functions as c_f


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Exactly the same as torch.utils.data.ConcatDataset
    except with a nice ```__repr__``` function.
    """

    def __repr__(self):
        extra_repr = f"len={str(self.__len__())}"
        return c_f.nice_repr(self, extra_repr, {"datasets": self.datasets})

from typing import List

from ..utils import common_functions as c_f
from .multiple_containers import MultipleContainers


class KeyEnforcer:
    """
    Makes sure containers have the specified keys.
    """

    def __init__(self, **kwargs: List[str]):
        """
        Arguments:
            **kwargs: A mapping from container name to a list of required
                keys for that container.
        """
        self.requirements = kwargs

    def check(self, containers: MultipleContainers):
        """
        Compares the input containers' keys to ```self.requirements```.
        Raises ```KeyError``` if there is a mismatch.
        Arguments:
            containers: The containers to check.
        """
        for k, required_keys in self.requirements.items():
            container_keys = list(containers[k].keys())
            r_c_diff = c_f.list_diff(required_keys, container_keys)
            c_r_diff = c_f.list_diff(container_keys, required_keys)
            error_msg = ""
            if len(r_c_diff) > 0:
                error_msg += (
                    f"The {k} container is missing the following keys: {r_c_diff}. "
                )

            if len(c_r_diff) > 0:
                error_msg += (
                    f"The {k} container has the following unallowed keys: {c_r_diff}."
                )

            if error_msg != "":
                raise KeyError(error_msg)

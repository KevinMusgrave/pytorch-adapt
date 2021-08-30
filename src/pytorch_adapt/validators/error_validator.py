from ..utils import common_functions as c_f
from .accuracy_validator import AccuracyValidator


class ErrorValidator(AccuracyValidator):
    def compute_score(self, src_val):
        return 1 - super().compute_score(src_val)

    @property
    def maximize(self):
        return False

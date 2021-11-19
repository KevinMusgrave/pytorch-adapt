from .accuracy_validator import AccuracyValidator


class ErrorValidator(AccuracyValidator):
    """
    Returns ```-(1-accuracy)```
    """

    def compute_score(self, src_val):
        return -(1 - super().compute_score(src_val))

from .diversity_validator import DiversityValidator
from .entropy_validator import EntropyValidator
from .multiple_validators import MultipleValidators


class IMValidator(MultipleValidators):
    """
    The sum of [EntropyValidator][pytorch_adapt.validators.entropy_validator]
    and [DiversityValidator][pytorch_adapt.validators.diversity_validator]
    """

    def __init__(self, weights=None, **kwargs):
        validators = {
            "entropy": EntropyValidator(**kwargs),
            "diversity": DiversityValidator(**kwargs),
        }
        super().__init__(validators=validators, weights=weights, **kwargs)

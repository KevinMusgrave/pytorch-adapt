from .diversity_validator import DiversityValidator
from .entropy_validator import EntropyValidator
from .multiple_validators import MultipleValidators


class EntropyDiversityValidator(MultipleValidators):
    def __init__(self, **kwargs):
        validators = {"entropy": EntropyValidator(), "diversity": DiversityValidator()}
        super().__init__(validators=validators, **kwargs)

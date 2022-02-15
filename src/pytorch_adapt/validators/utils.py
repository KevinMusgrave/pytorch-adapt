import numpy as np

from ..utils import common_functions as c_f
from .base_validator import BaseValidator
from .score_history import ScoreHistory


def max_normalizer(raw_score_history):
    return raw_score_history / np.abs(np.nanmax(raw_score_history))


def get_validation_score(validator, collected_data, epoch=None):
    kwargs = c_f.filter_kwargs(collected_data, validator.required_data)
    if isinstance(validator, BaseValidator):
        return validator(**kwargs)
    elif isinstance(validator, ScoreHistory):
        return validator(epoch, **kwargs)
    else:
        raise TypeError(
            "validator must be an instance of BaseValidator or ScoreHistory"
        )

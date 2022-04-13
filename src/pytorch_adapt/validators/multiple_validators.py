import itertools
from typing import Dict, List, Tuple, Union

from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f
from .base_validator import BaseValidator


class MultipleValidators(BaseValidator):
    """
    Wraps multiple validators and returns the sum of their scores.
    """

    def __init__(
        self,
        validators: Union[List["BaseValidator"], Dict[str, "BaseValidator"]],
        weights: Union[List[float], Dict[str, float]] = None,
        return_sub_scores=False,
        **kwargs,
    ):
        """
        Arguments:
            validators: A list of validators or a dictionary mapping from strings to validators.
            weights: A list of floats or a dictionary mapping from validator names to floats.
                If ```None```, then the validators will be equally weighted.
            return_sub_scores: If ```True```, then return the score of each validator,
                in addition to their summed value.
        """
        super().__init__(**kwargs)
        self.validators = c_f.enumerate_to_dict(validators)
        self.weights = c_f.default(weights, {k: 1 for k in self.validators.keys()})
        self.weights = c_f.enumerate_to_dict(self.weights)
        if self.validators.keys() != self.weights.keys():
            raise KeyError("validator keys and weight keys must be the same")
        self.return_sub_scores = return_sub_scores
        pml_cf.add_to_recordable_attributes(self, list_of_names=["weights"])

    def _required_data(self):
        output = [v.required_data for v in self.validators.values()]
        output = list(itertools.chain(*output))
        return list(set(output))

    def compute_score(self):
        pass

    def __call__(self, **kwargs) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Returns:
            The sum of the validator scores. If ```self.return_sub_scores``` then
            it also returns a dictionary containing each validator's weighted score.
        """
        kwargs = self.kwargs_check(kwargs)
        outputs = {}
        for k, v in self.validators.items():
            score = v(**c_f.filter_kwargs(kwargs, v.required_data))
            outputs[k] = score * self.weights[k]
        final = sum(outputs.values())
        if self.return_sub_scores:
            return final, outputs
        return final

    def __repr__(self):
        return c_f.nice_repr(self, self.extra_repr(), self.validators)

    def extra_repr(self):
        x = super().extra_repr()
        x += f"\n{c_f.extra_repr(self, ['weights'])}"
        return x

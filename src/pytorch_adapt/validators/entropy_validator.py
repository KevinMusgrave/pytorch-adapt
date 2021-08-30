from ..layers import EntropyLoss
from .base_validator import BaseValidator


class EntropyValidator(BaseValidator):
    def compute_score(self, target_train):
        return EntropyLoss()(target_train["logits"]).item()

    @property
    def maximize(self):
        return False

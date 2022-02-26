from ...validators import utils as val_utils


class IgniteValHookWrapper:
    def __init__(self, validator, logger=None):
        self.validator = validator
        self.logger = logger

    def __call__(self, epoch, **kwargs):
        score = val_utils.call_val_hook(self.validator, kwargs, epoch)
        if self.logger:
            self.logger.add_validation({"val_hook": self.validator}, epoch)

    @property
    def required_data(self):
        return self.validator.required_data

    def state_dict(self):
        return self.validator.state_dict()

    def load_state_dict(self, state_dict):
        self.validator.load_state_dict(state_dict)

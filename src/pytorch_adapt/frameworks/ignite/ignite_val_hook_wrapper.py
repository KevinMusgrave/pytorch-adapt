from ...validators import utils as val_utils


class IgniteValHookWrapper:
    def __init__(self, validator, saver=None, logger=None):
        self.validator = validator
        self.saver = saver
        self.logger = logger

    def __call__(self, epoch, **kwargs):
        score = val_utils.get_validation_score(self.validator, kwargs, epoch)
        if self.saver:
            self.saver.save_validator(self.validator, "val_hook")
        if self.logger:
            self.logger.add_validation({"val_hook": self.validator}, epoch)

    @property
    def required_data(self):
        return self.validator.required_data

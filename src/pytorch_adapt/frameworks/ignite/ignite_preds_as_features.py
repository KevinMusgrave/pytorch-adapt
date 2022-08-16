from .. import utils as f_utils
from .ignite import Ignite


# This is for the case where
# G outputs preds
# C is Identity()
# In other words, the "features" are softmaxed logits
class IgnitePredsAsFeatures(Ignite):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            val_output_dict_fn=f_utils.create_output_dict_preds_as_features,
            **kwargs
        )

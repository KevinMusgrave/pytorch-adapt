from .. import utils as f_utils
from .ignite import Ignite


# preds is sigmoid instead of softmax
class IgniteMultiLabelClassification(Ignite):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            val_output_dict_fn=f_utils.create_output_dict_multilabel_classification,
            **kwargs
        )

from ...utils import common_functions as c_f
from .. import utils as f_utils
from .ignite import Ignite


# This is for the case where
# G outputs preds
# C is Identity()
# In other words, the "features" are softmaxed logits
class IgnitePredsAsFeatures(Ignite):
    def get_collector_step(self, inference):
        def collector_step(engine, batch):
            batch = c_f.batch_to_device(batch, self.device)
            return f_utils.collector_step(
                inference, batch, f_utils.create_output_dict_preds_as_features
            )

        return collector_step

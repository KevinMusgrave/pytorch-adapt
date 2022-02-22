from pytorch_adapt.adapters import DANN
from pytorch_adapt.frameworks.ignite import Ignite
from pytorch_adapt.frameworks.ignite.loggers import IgniteRecordKeeperLogger

from .. import TEST_FOLDER
from .utils import get_datasets, get_gcd


def get_dann(
    inference_fn=None, log_freq=50, validator=None, val_hooks=None, checkpoint_fn=None
):
    models = get_gcd()
    dann = DANN(models=models, inference_fn=inference_fn)
    logger = IgniteRecordKeeperLogger(folder=TEST_FOLDER)
    datasets = get_datasets()
    return (
        Ignite(
            dann,
            validator=validator,
            val_hooks=val_hooks,
            checkpoint_fn=checkpoint_fn,
            logger=logger,
            log_freq=log_freq,
        ),
        datasets,
    )

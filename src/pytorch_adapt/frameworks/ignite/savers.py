import os
from abc import ABC, abstractmethod

import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ...utils import common_functions as c_f
from ...validators import ScoreHistories
from ignite.handlers import ModelCheckpoint


class BaseSaver(ABC):
    def __init__(self, folder):
        self.folder = folder
        c_f.makedir_if_not_there(self.folder)

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ValidatorSaver(BaseSaver):
    def save(self, validator, prefix=""):
        if is_multiple_histories(validator):
            self.save_multiple(validator, prefix)
        else:
            self.save_single(validator, prefix)

    def load(self, validator, prefix=""):
        if is_multiple_histories(validator):
            self.load_multiple(validator, prefix)
        else:
            self.load_single(validator, prefix)

    def save_single(self, validator, prefix=""):
        # cast best_epoch from int64 to regular int so it can be json-parsed
        best_epoch = validator.best_epoch
        if best_epoch is not None:
            best_epoch = int(best_epoch)
        output = {
            "best_score": validator.best_score,
            "best_epoch": best_epoch,
        }
        filename = c_f.class_as_prefix(validator, "best", prefix=prefix)
        c_f.save_json(output, self.folder, f"{filename}.json")
        for name in self.saveable_attrs:
            filename = c_f.class_as_prefix(validator, name, prefix=prefix)
            c_f.save_npy(getattr(validator, name), self.folder, f"{filename}.npy")

    def load_single(self, validator, prefix=""):
        for name in self.saveable_attrs:
            filename = c_f.class_as_prefix(validator, name, prefix=prefix)
            setattr(validator, name, c_f.load_npy(self.folder, f"{filename}.npy"))

    def save_multiple(self, validator, prefix=""):
        for k, v in validator.histories.items():
            child_prefix = c_f.class_as_prefix(validator, k, prefix=prefix)
            self.save_single(v, child_prefix)
        self.save_single(validator, prefix)

    def load_multiple(self, validator, prefix=""):
        for k, v in validator.histories.items():
            child_prefix = c_f.class_as_prefix(validator, k, prefix=prefix)
            self.load_single(v, child_prefix)
        self.load_single(validator, prefix)

    @property
    def saveable_attrs(self):
        return ["raw_score_history", "score_history", "epochs"]


def getAdapterCheckpointFn(adapter, **kwargs):
    dirname = kwargs.pop("dirname")
    dict_to_save = {}
    for container in ["models", "optimizers", "lr_schedulers", "misc"]:
        dict_to_save.update({f"{container}_{k}":v for k,v in getattr(adapter, container).items()})
    
    handler = ModelCheckpoint(dirname, "adapter", **kwargs)

    def fn(engine):
        handler(engine, dict_to_save)

    return fn



def is_multiple_histories(validator):
    return isinstance(validator, ScoreHistories)

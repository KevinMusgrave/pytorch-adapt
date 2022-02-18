import os
from abc import ABC, abstractmethod

import torch
from ignite.handlers import ModelCheckpoint
from pytorch_metric_learning.utils import common_functions as pml_cf

from ...utils import common_functions as c_f
from ...validators import ScoreHistories


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


def get_engine_checkpoint_fn(**kwargs):
    handler = ModelCheckpoint(**kwargs)

    def fn(engine):
        handler(engine, {"engine": engine})

    return fn


def get_adapter_checkpoint_fn(**kwargs):
    def fn_creator(adapter, score_function):
        dict_to_save = {}
        for container in ["models", "optimizers", "lr_schedulers", "misc"]:
            dict_to_save.update(
                {f"{container}_{k}": v for k, v in getattr(adapter, container).items()}
            )
        handler = ModelCheckpoint(score_function=score_function, **kwargs)

        def fn(engine):
            handler(engine, dict_to_save)

        return fn

    return fn_creator


def get_validator_checkpoint_fn(**kwargs):
    def fn_creator(validator):
        handler = ModelCheckpoint(**kwargs)

        def fn(engine):
            handler(engine, {"validator": validator})

        return fn

    return fn_creator


class CheckpointFn:
    def __init__(
        self,
        common_kwargs=None,
        engine_kwargs=None,
        adapter_kwargs=None,
        validator_kwargs=None,
    ):
        [
            common_kwargs,
            engine_kwargs,
            adapter_kwargs,
            validator_kwargs,
        ] = c_f.many_default(
            [common_kwargs, engine_kwargs, adapter_kwargs, validator_kwargs],
            [{"filename_prefix": ""}, {}, {}, {}],
        )
        self.engine_fn = get_engine_checkpoint_fn(**common_kwargs, **engine_kwargs)
        self.adapter_fn = get_adapter_checkpoint_fn(**common_kwargs, **adapter_kwargs)
        self.validator_fn = get_validator_checkpoint_fn(
            **common_kwargs, **validator_kwargs
        )


def is_multiple_histories(validator):
    return isinstance(validator, ScoreHistories)

import os
from abc import ABC, abstractmethod

import torch
from ignite.handlers import ModelCheckpoint
from pytorch_metric_learning.utils import common_functions as pml_cf

from ...utils import common_functions as c_f
from ...validators import ScoreHistories


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filename_pattern=None, **kwargs):
        super().__init__(**kwargs)
        self.filename_pattern = filename_pattern


def get_engine_checkpoint_fn(**kwargs):
    handler = CustomModelCheckpoint(**kwargs)

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
        handler = CustomModelCheckpoint(score_function=score_function, **kwargs)

        def fn(engine):
            handler(engine, dict_to_save)

        return fn

    return fn_creator


def get_validator_checkpoint_fn(**kwargs):
    def fn_creator(validator):
        handler = CustomModelCheckpoint(**kwargs)

        def fn(engine):
            handler(engine, {"validator": validator})

        return fn

    return fn_creator


def global_step_transform(engine, _):
    return engine.state.epoch


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
            [{}, {}, {}, {}],
        )
        common_kwargs = {
            "filename_prefix": "",
            "global_step_transform": global_step_transform,
            "filename_pattern": "{filename_prefix}{name}_{global_step}.pt",
            **common_kwargs,
        }
        adapter_kwargs = {
            "filename_prefix": "adapter_",
            "filename_pattern": "{filename_prefix}{global_step}.pt",
            **adapter_kwargs,
        }
        self.engine_fn = get_engine_checkpoint_fn(**{**common_kwargs, **engine_kwargs})
        self.adapter_fn = get_adapter_checkpoint_fn(
            **{**common_kwargs, **adapter_kwargs}
        )
        self.validator_fn = get_validator_checkpoint_fn(
            **{**common_kwargs, **validator_kwargs}
        )


def is_multiple_histories(validator):
    return isinstance(validator, ScoreHistories)

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


class EngineFn:
    def __init__(self, **kwargs):
        self.handler = CustomModelCheckpoint(**kwargs)
        self.dirname = kwargs["dirname"]
        self.dict_to_save = {}

    def __call__(self, engine):
        self.dict_to_save = {"engine": engine}
        self.handler(engine, self.dict_to_save)

    def load(self, engine, filename):
        to_load = {"engine": engine}
        checkpoint = os.path.join(self.dirname, filename)
        self.handler.load_objects(to_load=to_load, checkpoint=checkpoint, strict=False)


def global_step_transform(engine, _):
    return engine.state.epoch


class CheckpointFnCreator:
    def __init__(self, **kwargs):
        self.kwargs = {
            "filename_prefix": "",
            "global_step_transform": global_step_transform,
            "filename_pattern": "{filename_prefix}{name}_{global_step}.{ext}",
            **kwargs,
        }

    def __call__(self, adapter, validator=None, val_hooks=None, **kwargs):
        self.handler = CustomModelCheckpoint(**{**self.kwargs, **kwargs})
        dict_to_save = {"adapter": adapter}
        if validator:
            dict_to_save["validator"] = validator
        if val_hooks:
            for i, v in enumerate(val_hooks):
                if not all(hasattr(v, x) for x in ["state_dict", "load_state_dict"]):
                    c_f.LOGGER.warning(
                        "val_hook has no state_dict or load_state_dict so it will not be saved"
                    )
                else:
                    dict_to_save[f"val_hook{i}"] = v

        def fn(engine):
            self.handler(engine, {"engine": engine, **dict_to_save})

        return fn

    def load_objects(self, *args, **kwargs):
        self.handler.load_objects(*args, **kwargs)

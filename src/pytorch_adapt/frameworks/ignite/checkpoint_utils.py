import glob
import os

from ignite.handlers import ModelCheckpoint

from ...adapters.utils import container_names
from ...utils import common_functions as c_f


def global_step_transform(engine, _):
    return engine.state.epoch


def val_hooks_to_dict(val_hooks):
    output = {}
    for i, v in enumerate(val_hooks):
        if not all(hasattr(v, x) for x in ["state_dict", "load_state_dict"]):
            c_f.LOGGER.warning(
                "val_hook has no state_dict or load_state_dict so it will not be saved or loaded"
            )
        else:
            output[f"val_hook{i}"] = v
    return output


def adapter_to_dict(adapter):
    return {k: getattr(adapter, k) for k in container_names()}


class CheckpointFnCreator:
    def __init__(self, **kwargs):
        self.kwargs = {
            "filename_prefix": "",
            "global_step_transform": global_step_transform,
            "filename_pattern": "{filename_prefix}{name}_{global_step}.{ext}",
            **kwargs,
        }
        # Create handler here in case needed by load_objects or last_checkpoint
        # before __call__ is used
        self.objs = ModelCheckpoint(**self.kwargs)

        # For saving self.objs. Only save the very latest (n_saved = 1)
        self.ckpter = ModelCheckpoint(**{**self.kwargs, "n_saved": 1})

    def __call__(self, adapter=None, validator=None, val_hooks=None, **kwargs):
        self.objs = ModelCheckpoint(**{**self.kwargs, **kwargs})
        dict_to_save = {}
        if adapter:
            dict_to_save.update(adapter_to_dict(adapter))
        if validator:
            dict_to_save["validator"] = validator
        if val_hooks:
            dict_to_save.update(val_hooks_to_dict(val_hooks))

        def fn(engine):
            self.objs(engine, {"engine": engine, **dict_to_save})
            self.ckpter(engine, {"checkpointer": self.objs})

        return fn

    def load_objects(self, to_load, checkpoint=None, global_step=None, **kwargs):
        # This can be simplified once this issue is resolved https://github.com/pytorch/ignite/issues/2480
        if global_step is not None:
            dirname = self.objs.save_handler.dirname
            filename_dict = {
                "filename_prefix": self.objs.filename_prefix,
                "name": "checkpoint",
                "ext": self.objs.ext,
                "score_name": self.objs.score_name,
                "global_step": global_step,
            }
            filename = self.objs.filename_pattern.format(**filename_dict)
            checkpoint = os.path.join(dirname, filename)

        to_load = {k: v for k, v in to_load.items() if v}
        self.objs.load_objects(to_load, str(checkpoint), **kwargs)

    def load_best_checkpoint(self, to_load):
        last_checkpoint = self.get_best_checkpoint()
        self.load_objects(to_load, last_checkpoint)

    def get_best_checkpoint(self):
        if self.objs.last_checkpoint:
            return self.objs.last_checkpoint

        ckpter_last_checkpoint = self.ckpter.last_checkpoint
        if not ckpter_last_checkpoint:
            files = glob.glob(
                os.path.join(self.ckpter.save_handler.dirname, "*checkpointer*.pt")
            )
            if len(files) > 1:
                raise ValueError("there should only be 1 matching checkpointer file")
            ckpter_last_checkpoint = files[0]

        self.ckpter.load_objects(
            {"checkpointer": self.objs}, str(ckpter_last_checkpoint)
        )
        return self.objs.last_checkpoint

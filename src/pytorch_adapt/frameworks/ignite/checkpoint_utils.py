import os

from ignite.handlers import ModelCheckpoint

from ...utils import common_functions as c_f


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filename_pattern=None, **kwargs):
        super().__init__(**kwargs)
        self.filename_pattern = filename_pattern


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
        self.handler = CustomModelCheckpoint(**self.kwargs)

        # For saving self.handler. Only save the very latest (n_saved = 1)
        self.self_handler = CustomModelCheckpoint(**{**self.kwargs, "n_saved": 1})

    def __call__(self, adapter=None, validator=None, val_hooks=None, **kwargs):
        self.handler = CustomModelCheckpoint(**{**self.kwargs, **kwargs})
        dict_to_save = {}
        if adapter:
            dict_to_save["adapter"] = adapter
        if validator:
            dict_to_save["validator"] = validator
        if val_hooks:
            dict_to_save.update(val_hooks_to_dict(val_hooks))

        def fn(engine):
            self.handler(engine, {"engine": engine, **dict_to_save})
            self.self_handler(engine, {"checkpointer": self.handler})

        return fn

    def load_objects(self, to_load, checkpoint=None, global_step=None, **kwargs):
        # This can be simplified once this issue is resolved https://github.com/pytorch/ignite/issues/2480
        if global_step is not None:
            dirname = self.handler.save_handler.dirname
            filename_dict = {
                "filename_prefix": self.handler.filename_prefix,
                "name": "checkpoint",
                "ext": self.handler.ext,
                "score_name": self.handler.score_name,
                "global_step": global_step,
            }
            filename = self.handler.filename_pattern.format(**filename_dict)
            checkpoint = os.path.join(dirname, filename)

        to_load = {k: v for k, v in to_load.items() if v}
        self.handler.load_objects(to_load, checkpoint, **kwargs)

    @property
    def last_checkpoint(self):
        return self.handler.last_checkpoint

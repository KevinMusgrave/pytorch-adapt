import os
from abc import ABC, abstractmethod

import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from . import common_functions as c_f


class BaseSaver(ABC):
    def __init__(self, folder):
        self.folder = folder

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ContainerSaver(BaseSaver):
    def save(self, container, epoch, best_epoch, prefix=""):
        suffixes = [str(epoch)]
        if epoch == best_epoch:
            suffixes.append("best")
        for suffix in suffixes:
            for k, v in container.items():
                if not c_f.has_state_dict(v):
                    continue
                filename = c_f.class_as_prefix(
                    container, k, prefix=prefix, suffix=suffix
                )
                c_f.save_torch_module(v, self.folder, f"{filename}.pth")

    def load(self, container, suffix, prefix=""):
        for k, v in container.items():
            if not c_f.has_state_dict(v):
                continue
            if suffix == "latest":
                basename = c_f.class_as_prefix(container, k, prefix=prefix)
                suffix, _ = pml_cf.latest_version(
                    self.folder, string_to_glob=f"{basename}*.pth"
                )
            filename = c_f.class_as_prefix(container, k, prefix=prefix, suffix=suffix)
            c_f.load_torch_module(v, self.folder, f"{filename}.pth")
        return suffix

    def delete(self, container, keep, prefix=""):
        to_keep = [(prefix, keep), (prefix, "best")]
        for k, v in container.items():
            if not c_f.has_state_dict(v):
                continue
            basename = c_f.class_as_prefix(container, k)
            c_f.delete_all_but(self.folder, basename, ".pth", to_keep=to_keep)

    def copy(self, container, from_suffix, to_suffix, prefix=""):
        c_f.LOGGER.info(f"Copying {from_suffix} to {to_suffix}")
        for k, v in container.items():
            if not c_f.has_state_dict(v):
                continue
            filenames = [
                c_f.class_as_prefix(container, k, prefix=prefix, suffix=x)
                for x in [from_suffix, to_suffix]
            ]
            filenames = [os.path.join(self.folder, f"{x}.pth") for x in filenames]
            c_f.copy_file(filenames[0], filenames[1])


class MultipleContainersSaver(BaseSaver):
    def __init__(self, container_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.container_saver = c_f.default(container_saver, ContainerSaver, kwargs)

    def save(self, container, epoch, best_epoch, prefix=""):
        # save each child container
        for v in container.values():
            self.container_saver.save(v, epoch, best_epoch, prefix)

    def load(self, container, suffix, prefix=""):
        # load each child container
        for v in container.values():
            self.container_saver.load(v, suffix, prefix)

    def delete(self, container, keep, prefix=""):
        for v in container.values():
            self.container_saver.delete(v, keep, prefix)

    def copy(self, container, from_suffix, to_suffix, prefix=""):
        for v in container.values():
            self.container_saver.copy(v, from_suffix, to_suffix, prefix)


class AdapterSaver(BaseSaver):
    def __init__(self, container_saver=None, keep_every_version=False, **kwargs):
        super().__init__(**kwargs)
        self.container_saver = c_f.default(
            container_saver, MultipleContainersSaver, kwargs
        )
        self.keep_every_version = keep_every_version

    def save(self, adapter, epoch, best_epoch, prefix=""):
        self.container_saver.save(adapter.containers, epoch, best_epoch, prefix=prefix)
        if not self.keep_every_version:
            self.container_saver.delete(adapter.containers, keep=epoch, prefix=prefix)
        if self.keep_every_version and epoch != best_epoch:
            self.container_saver.copy(
                adapter.containers, best_epoch, "best", prefix=prefix
            )

    def load(self, adapter, suffix, prefix=""):
        self.container_saver.load(adapter.containers, suffix, prefix=prefix)


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


class IgniteSaver(BaseSaver):
    def filename(self):
        return c_f.full_path(self.folder, "ignite_engine.pth")

    def save(self, engine):
        torch.save(engine.state_dict(), self.filename())

    def load(self, engine):
        engine.load_state_dict(torch.load(self.filename()))


class Saver:
    def __init__(
        self,
        adapter_saver=None,
        validator_saver=None,
        stat_getter_saver=None,
        ignite_saver=None,
        **kwargs,
    ):
        self.adapter_saver = c_f.default(adapter_saver, AdapterSaver, kwargs)
        self.validator_saver = c_f.default(validator_saver, ValidatorSaver, kwargs)
        self.stat_getter_saver = c_f.default(stat_getter_saver, ValidatorSaver, kwargs)
        self.ignite_saver = c_f.default(ignite_saver, IgniteSaver, kwargs)

    def save_adapter(self, adapter, epoch, best_epoch):
        c_f.LOGGER.info(f"Saving adapter to {self.adapter_saver.folder}")
        self.adapter_saver.save(adapter, epoch, best_epoch, "adapter")

    def load_adapter(self, adapter, suffix):
        c_f.LOGGER.info(f"Loading adapter from {self.adapter_saver.folder}")
        self.adapter_saver.load(adapter, suffix, "adapter")

    def save_validator(self, validator):
        c_f.LOGGER.info(f"Saving validator to {self.validator_saver.folder}")
        self.validator_saver.save(validator, "validator")

    def load_validator(self, validator):
        c_f.LOGGER.info(f"Loading validator from {self.validator_saver.folder}")
        self.validator_saver.load(validator, "validator")

    def save_stat_getter(self, stat_getter):
        c_f.LOGGER.info(f"Saving stat_getter to {self.stat_getter_saver.folder}")
        self.stat_getter_saver.save(stat_getter, "stat_getter")

    def load_stat_getter(self, stat_getter):
        c_f.LOGGER.info(f"Loading stat_getter from {self.stat_getter_saver.folder}")
        self.stat_getter_saver.load(stat_getter, "stat_getter")

    def save_ignite(self, engine):
        c_f.LOGGER.info(f"Saving Ignite engine to {self.ignite_saver.folder}")
        self.ignite_saver.save(engine)

    def load_ignite(self, engine):
        c_f.LOGGER.info(f"Loading Ignite engine from {self.ignite_saver.folder}")
        self.ignite_saver.load(engine)

    def load_all(
        self,
        adapter=None,
        validator=None,
        stat_getter=None,
        framework=None,
        suffix="latest",
    ):
        if adapter:
            self.load_adapter(adapter, suffix)
        if validator:
            self.load_validator(validator)
        if stat_getter:
            self.load_stat_getter(stat_getter)
        if framework:
            from ..frameworks.ignite import Ignite

            if isinstance(framework, Ignite):
                self.load_ignite(framework.trainer)


def is_multiple_histories(validator):
    # to avoid circular import
    from ..validators.score_history import ScoreHistories

    return isinstance(validator, ScoreHistories)

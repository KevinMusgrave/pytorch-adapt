import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ...hooks.base import BaseHook
from ...utils import common_functions as c_f
from ...weighters.base_weighter import BaseWeighter


class IgniteEmptyLogger:
    def add_training(self, *args, **kwargs):
        pass

    def add_validation(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


class IgniteRecordKeeperLogger:
    """
    Uses [record-keeper](https://github.com/KevinMusgrave/record-keeper)
    to record data tensorboard, csv, and sqlite.
    """

    def __init__(
        self,
        folder=None,
        tensorboard_writer=None,
        record_writer=None,
        attr_list_names=None,
    ):
        """
        Arguments:
            folder: path where records will be saved.
            tensorboard_writer:
            record_writer: a ```RecordWriter``` object (see record-keeper)
        """
        import record_keeper

        c_f.LOGGER.info(f"record_keeper version {record_keeper.__version__}")
        from record_keeper import RecordKeeper, RecordWriter
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_writer = c_f.default(
            tensorboard_writer,
            SummaryWriter,
            {"log_dir": folder, "max_queue": 1000000, "flush_secs": 30},
        )
        record_writer = c_f.default(record_writer, RecordWriter, {"folder": folder})
        attr_list_names = c_f.default(
            attr_list_names, pml_cf.list_of_recordable_attributes_list_names()
        )
        self.record_keeper = RecordKeeper(
            tensorboard_writer=tensorboard_writer,
            record_writer=record_writer,
            attributes_to_search_for=attr_list_names,
        )

    def add_training(self, engine):
        adapter = engine.state.adapter
        record_these = [
            ({"engine_output": engine.state.output}, {}),
            (
                adapter.optimizers,
                {
                    "parent_name": "optimizers",
                    "custom_attr_func": optimizer_attr_func,
                },
            ),
            ({"misc": adapter.misc}, {}),
            (
                {"hook": adapter.hook},
                {"recursive_types": [BaseHook, BaseWeighter, torch.nn.Module]},
            ),
        ]
        for record, kwargs in record_these:
            self.record_keeper.update_records(record, engine.state.iteration, **kwargs)

    def add_validation(self, data, epoch):
        self.record_keeper.update_records(data, epoch)

    def write(self, engine):
        self.record_keeper.save_records()
        self.record_keeper.tensorboard_writer.flush()


def optimizer_attr_func(optimizer):
    return {"lr": c_f.get_lr(optimizer)}

import csv
import glob
import os
import shutil
from pathlib import Path

from pytorch_adapt.frameworks.ignite import Ignite, savers
from pytorch_adapt.frameworks.ignite.loggers import IgniteRecordKeeperLogger
from pytorch_adapt.validators import (
    AccuracyValidator,
    EntropyValidator,
    MultipleValidators,
    ScoreHistories,
    ScoreHistory,
)

from .utils import get_datasets


# log files should be a mapping from csv file name, to number of columns in file
def run_adapter(cls, test_folder, adapter, log_files=None):
    saver = savers.Saver(folder=test_folder)
    datasets = get_datasets()
    validator = ScoreHistory(EntropyValidator())
    stat_getter = MultipleValidators(
        {
            "src_train": AccuracyValidator(key_map={"src_train": "src_val"}),
            "src_val": AccuracyValidator(),
        },
    )
    stat_getter = ScoreHistories(stat_getter)
    logger = IgniteRecordKeeperLogger(folder=test_folder)
    adapter = Ignite(
        adapter,
        validator=validator,
        stat_getter=stat_getter,
        saver=saver,
        logger=logger,
        log_freq=1,
    )
    adapter.run(
        datasets=datasets,
    )
    if log_files:
        all_logs = glob.glob(os.path.join(test_folder, "*.csv"))
        all_logs = [Path(x).name for x in all_logs]
        log_files = {f"{k}.csv": v for k, v in log_files.items()}
        cls.assertTrue(sorted(list(log_files.keys())) == sorted(all_logs))
        for k, v in log_files.items():
            with open(os.path.join(test_folder, k), "r") as f:
                reader = csv.reader(f)
                row = next(reader)
            row.remove("~iteration~")
            cls.assertTrue(set(row) == v)

    shutil.rmtree(test_folder)
    del saver
    del datasets
    del logger
    del adapter

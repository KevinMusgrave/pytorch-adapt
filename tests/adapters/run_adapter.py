import csv
import glob
import os
import shutil
from pathlib import Path

import torch

from pytorch_adapt.frameworks.ignite import (
    CheckpointFnCreator,
    Ignite,
    IgniteValHookWrapper,
)
from pytorch_adapt.frameworks.ignite.loggers import IgniteRecordKeeperLogger
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import (
    AccuracyValidator,
    EntropyValidator,
    MultipleValidators,
    ScoreHistories,
    ScoreHistory,
)

from .. import TEST_DEVICE
from .utils import get_datasets


# log files should be a mapping from csv file name, to number of columns in file
def run_adapter(cls, test_folder, adapter, log_files=None, inference_fn=None):
    checkpoint_fn = CheckpointFnCreator(dirname=test_folder)
    logger = IgniteRecordKeeperLogger(folder=test_folder)
    datasets = get_datasets()
    validator = ScoreHistory(EntropyValidator())
    val_hook = MultipleValidators(
        {
            "src_train": AccuracyValidator(key_map={"src_train": "src_val"}),
            "src_val": AccuracyValidator(),
        },
    )
    val_hook = ScoreHistories(val_hook)
    val_hook = IgniteValHookWrapper(val_hook, logger=logger)
    adapter = Ignite(
        adapter,
        validator=validator,
        val_hooks=[val_hook],
        checkpoint_fn=checkpoint_fn,
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

    if inference_fn:
        for split in ["src", "target"]:
            x = datasets[f"{split}_train"][0]
            [x, domain] = c_f.extract(x, [f"{split}_imgs", f"{split}_domain"])
            x = x.unsqueeze(0).to(TEST_DEVICE)
            domain = torch.tensor([domain], device=TEST_DEVICE)
            correct = inference_fn(
                x=x,
                domain=domain,
                models=adapter.adapter.models,
                misc=adapter.adapter.misc,
            )
            output = adapter.adapter.inference(x, domain)
            cls.assertTrue(correct.keys() == output.keys())
            for k, v in correct.items():
                cls.assertTrue(torch.equal(v, output[k]))

    shutil.rmtree(test_folder)
    del checkpoint_fn
    del datasets
    del logger
    del adapter

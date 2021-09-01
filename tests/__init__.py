import os
import sys

import torch

sys.path.insert(0, "src")
import logging

import pytorch_adapt
from pytorch_adapt.utils import common_functions as c_f

logging.basicConfig()
logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.INFO)


c_f.LOGGER.info(
    "testing pytorch_adapt version {} with pytorch version {}".format(
        pytorch_adapt.__version__, torch.__version__
    )
)

dtypes_from_environ = os.environ.get("TEST_DTYPES", "float16,float32,float64").split(
    ","
)
device_from_environ = os.environ.get("TEST_DEVICE", "cuda")
TEST_FOLDER = os.environ.get("TEST_FOLDER", "zzz_pytorch_adapt_test_folder")
DATASET_FOLDER = os.environ.get("DATASET_FOLDER", "zzz_pytorch_adapt_test_folder")
RUN_DATASET_TESTS = os.environ.get("RUN_DATASET_TESTS", False)

TEST_DTYPES = [getattr(torch, x) for x in dtypes_from_environ]
TEST_DEVICE = torch.device(device_from_environ)

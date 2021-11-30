import pytorch_lightning as pl
import torch

from ...containers import Optimizers
from .. import utils as f_utils


def set_adapter_optimizers_to_pl(adapter, pl_optimizers):
    if isinstance(adapter.optimizers, Optimizers):
        keys = adapter.optimizers.keys()
        adapter.optimizers = {k: v for k, v in zip(keys, pl_optimizers)}


class Lightning(pl.LightningModule):
    def __init__(self, adapter, validator=None):
        super().__init__()
        self.models = torch.nn.ModuleDict(adapter.models)
        self.misc = torch.nn.ModuleDict(adapter.misc)
        adapter.models = self.models
        adapter.misc = self.misc
        self.validator = validator
        self.adapter = adapter
        self.automatic_optimization = False

    def forward(self, x, domain=None):
        return self.adapter.inference(x, domain=domain)

    def training_step(self, batch, batch_idx):
        set_adapter_optimizers_to_pl(self.adapter, self.optimizers())
        losses = self.adapter.training_step(
            batch,
            custom_backward=self.manual_backward,
        )
        for k, v in losses.items():
            self.log(k, v)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return f_utils.collector_step(self, batch, f_utils.create_output_dict)

    def configure_optimizers(self):
        optimizers = list(self.adapter.optimizers.values())
        lr_schedulers = []
        for interval in ["epoch", "step"]:
            for v in self.adapter.lr_schedulers.filter_by_scheduler_type(
                f"per_{interval}"
            ):
                lr_schedulers.append({"lr_scheduler": v, "interval": interval})
        return optimizers, lr_schedulers

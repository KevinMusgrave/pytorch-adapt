import pytorch_lightning as pl
import torch


class Lightning(pl.LightningModule):
    def __init__(self, adapter, validator=None):
        super().__init__()
        self.models = torch.nn.ModuleDict(adapter.models)
        self.misc = torch.nn.ModuleDict(adapter.misc)
        del adapter.models
        del adapter.misc
        self.optimizer_keys = sorted(adapter.optimizers.keys())
        self.validator = validator
        self.adapter = adapter
        self.automatic_optimization = False

    def forward(self, x, domain=None):
        return self.adapter.inference(x, domain=domain)

    def training_step(self, batch, batch_idx):
        optimizers = {k: v for k, v in zip(self.optimizer_keys, self.optimizers())}
        losses = self.adapter.training_step(batch, models=self.models, misc=self.misc)

    def configure_optimizers(self):
        return [self.adapter.optimizers[k] for k in self.optimizer_keys]

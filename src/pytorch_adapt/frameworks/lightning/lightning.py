import pytorch_lightning as pl
import torch


class Lightning(pl.LightningModule):
    def __init__(self, adapter, validator=None):
        super().__init__()
        self.models = torch.nn.ModuleDict(adapter.models)
        self.misc = torch.nn.ModuleDict(adapter.misc)
        del adapter.models
        del adapter.misc
        self.validator = validator
        self.adapter = adapter
        self.automatic_optimization = False

    def forward(self, x, domain=None):
        return self.adapter.inference(x, domain=domain)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        print(self.adapter.hook)
        losses = self.adapter.training_step(batch, models=self.models, misc=self.misc)

    def configure_optimizers(self):
        keys = sorted(self.adapter.optimizers.keys())
        return [self.adapter.optimizers[k] for k in keys]

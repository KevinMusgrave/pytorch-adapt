import pytorch_lightning as pl
import torch


class Lightning(pl.LightningModule):
    def __init__(self, adapter, validator=None):
        super().__init__()
        self.adapter = adapter
        self.models = torch.nn.ModuleDict(adapter.models)
        self.validator = validator
        self.automatic_optimization = False

    def forward(self, x, domain=None):
        return self.adapter.inference(x, domain=domain)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        losses = self.adapter.training_step(batch)

    def configure_optimizers(self):
        keys = sorted(self.adapter.optimizers.keys())
        return [self.adapter.optimizers[k] for k in keys]

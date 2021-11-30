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
        losses = self.adapter.training_step(
            batch,
            models=self.models,
            optimizers=optimizers,
            misc=self.misc,
            custom_backward=self.manual_backward,
        )
        for k, v in losses.items():
            self.log(k, v)

    # def validation_step(self, batch, batch_idx, dataloader_idx):
    #     # batch is 1/num_gpus big
    #     dataset = curr_dataloader.dataset
    #     domain = f_utils.get_domain(dataset)
    #     f_utils.collector_step(inference, batch, name)

    #     return collector_step

    # def get_collector(self, dataset):

    #     return self.get_collector_step(inference, domain)

    def configure_optimizers(self):
        optimizers = [self.adapter.optimizers[k] for k in self.optimizer_keys]
        lr_schedulers = []
        for interval in ["epoch", "step"]:
            for v in self.adapter.lr_schedulers.filter_by_scheduler_type(
                f"per_{interval}"
            ):
                lr_schedulers.append({"lr_scheduler": v, "interval": interval})
        del self.adapter.optimizers
        del self.adapter.lr_schedulers
        return optimizers, lr_schedulers

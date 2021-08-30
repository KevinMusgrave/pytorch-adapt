import os

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan

from ...datasets import DataloaderCreator
from ...utils import common_functions as c_f
from . import utils as i_g
from .loggers import IgniteEmptyLogger


class Ignite:
    def __init__(self, adapter, logger=None, log_freq=50, with_pbars=True):
        self.adapter = adapter
        self.logger = c_f.default(logger, IgniteEmptyLogger, {})
        self.log_freq = log_freq
        self.with_pbars = with_pbars
        self.engine_init()
        self.dist_init_done = False

    def training_step(self, engine, batch):
        device = idist.device()
        batch = c_f.batch_to_device(batch, device)
        return self.adapter.training_step(batch, device, self)

    def before_training_starts(self, engine):
        self.adapter.before_training_starts(self)

    def engine_init(self):
        self.trainer = Engine(self.training_step)
        self.trainer.state.adapter = self.adapter
        self.labeled_collector = Engine(
            self.get_labeled_collector_step(self.adapter.inference)
        )
        self.unlabeled_collector = Engine(
            self.get_unlabeled_collector_step(self.adapter.inference)
        )
        i_g.do_for_all_engines(self, i_g.set_engine_logger)
        self.register_event_handlers()

    def register_event_handlers(self):
        if self.with_pbars:
            pbars = i_g.do_for_all_engines(self, i_g.attach_pbar)
        i_g.register(self.trainer, Events.STARTED, self.before_training_starts)
        i_g.register(
            self.trainer, Events.EPOCH_STARTED, self.set_to_train(self.adapter.models)
        )
        iteration_complete = [
            i_g.step_lr_schedulers(self.adapter.lr_schedulers, "per_step"),
            TerminateOnNan(),
        ]
        if self.with_pbars:
            iteration_complete.append(i_g.pbar_print_losses(pbars["trainer"]))
        i_g.register(self.trainer, Events.ITERATION_COMPLETED, *iteration_complete)
        i_g.register(
            self.trainer,
            Events.ITERATION_COMPLETED(every=self.log_freq),
            self.logger.add_training,
        )
        i_g.register(
            self.trainer,
            Events.EPOCH_COMPLETED,
            i_g.step_lr_schedulers(self.adapter.lr_schedulers, "per_epoch"),
            self.logger.write,
            i_g.zero_grad(self.adapter),
        )
        i_g.register(
            self.labeled_collector,
            Events.EPOCH_STARTED,
            self.set_to_eval(self.adapter.models),
        )
        i_g.register(
            self.unlabeled_collector,
            Events.EPOCH_STARTED,
            self.set_to_eval(self.adapter.models),
        )

    def dist_init(self, *args, **kwargs):
        if not self.dist_init_done:
            self.adapter.models.apply(i_g.auto_model(*args, **kwargs))
            self.adapter.optimizers.apply(idist.auto_optim)
            self.dist_init_done = True

    def get_training_length(self):
        max_epochs = self.trainer.state.max_epochs
        max_iters = max_epochs * self.trainer.state.epoch_length
        return max_epochs, max_iters

    def get_progress(self):
        _, max_iters = self.get_training_length()
        return float(self.get_iteration()) / max_iters

    def get_iteration(self):
        return self.trainer.state.iteration

    def get_epoch_length(self):
        return self.trainer.state.epoch_length

    def get_all_outputs(self, dataloader, split_name):
        dataloaders = {split_name: dataloader}
        return i_g.collect_from_dataloaders(self, dataloaders, [split_name])

    def run(
        self,
        datasets,
        dataloader_creator=None,
        validator=None,
        stat_getter=None,
        validation_interval=1,
        patience=10,
        saver=None,
        resume=None,
        check_initial_accuracy=False,
        **trainer_kwargs,
    ):
        dataloader_creator = c_f.default(dataloader_creator, DataloaderCreator())
        dataloaders = dataloader_creator(**datasets)

        self.dist_init()

        if validator:
            max_epochs = trainer_kwargs.get("max_epochs", 1)
            i_g.add_validation_runner(
                self,
                dataloaders,
                validator,
                stat_getter,
                validation_interval,
                max_epochs,
                saver,
                self.logger,
                check_initial_accuracy,
            )

            self.trainer.add_event_handler(
                Events.EPOCH_STARTED, i_g.early_stopper(patience, validator)
            )

        if saver:
            if not validator:
                self.trainer.add_event_handler(
                    Events.EPOCH_COMPLETED,
                    i_g.save_adapter_without_validator(saver, self.adapter),
                )
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, saver.save_ignite)

        if resume is not None:
            if resume != "latest":
                raise ValueError("Only 'latest' resume is currently supported")
            if not saver:
                raise ValueError("To resume, a Saver must be provided")
            saver.load_all(
                adapter=self.adapter,
                validator=validator,
                stat_getter=stat_getter,
                framework=self,
                suffix=resume,
            )
            i_g.resume_checks(validator, stat_getter, self)

        if not i_g.is_done(self.trainer, **trainer_kwargs):
            self.trainer.run(dataloaders["train"], **trainer_kwargs)

        if validator:
            return validator.best_score, validator.best_epoch

        return None, None

    def evaluate_best_model(
        self, datasets, validator, saver, epoch, dataloader_creator=None
    ):
        c_f.LOGGER.info("***EVALUATING BEST MODEL***")
        dataloader_creator = c_f.default(dataloader_creator, DataloaderCreator())
        dataloaders = dataloader_creator(**datasets)
        saver.load_adapter(self.adapter, "best")
        collected_data = i_g.collect_from_dataloaders(
            self, dataloaders, validator.required_data
        )
        return i_g.get_validation_score(collected_data, validator, epoch)

    def get_x_collector_step(self, inference, name):
        def collector_step(engine, batch):
            device = idist.device()
            with torch.no_grad():
                batch = c_f.batch_to_device(batch, device)
                features, logits = inference(
                    batch[f"{name}_imgs"], domain=batch[f"{name}_domain"]
                )
            output = {
                "features": features,
                "logits": logits,
                "preds": torch.softmax(logits, dim=1),
                "domain": batch[f"{name}_domain"],
            }
            labels_key = f"{name}_labels"
            if labels_key in batch:
                output["labels"] = batch[labels_key]
            return output

        return collector_step

    def get_labeled_collector_step(self, inference):
        return self.get_x_collector_step(inference, "src")

    def get_unlabeled_collector_step(self, inference):
        return self.get_x_collector_step(inference, "target")

    def get_collector(self, dataset):
        dataset_output = dataset[0]
        return (
            self.labeled_collector
            if len(dataset_output) == 4
            else self.unlabeled_collector
        )

    def set_to_train(self, models):
        def handler(engine):
            c_f.LOGGER.info("Setting models to train() mode")
            models.train()

        return handler

    def set_to_eval(self, models):
        def handler(engine):
            c_f.LOGGER.info("Setting models to eval() mode")
            models.eval()

        return handler

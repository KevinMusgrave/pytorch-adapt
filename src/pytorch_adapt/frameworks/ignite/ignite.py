import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan

from ...datasets import DataloaderCreator
from ...utils import common_functions as c_f
from .. import utils as f_utils
from . import utils as i_g
from .loggers import IgniteEmptyLogger


class Ignite:
    """
    Wraps an [Adapter](../../adapters/index.md) and takes
    care of validation, model saving, etc. by using
    the event handler system of PyTorch Ignite.
    """

    def __init__(
        self,
        adapter,
        validator=None,
        stat_getter=None,
        saver=None,
        logger=None,
        val_data_hook=None,
        log_freq=50,
        with_pbars=True,
        device=None,
        auto_dist=True,
    ):
        """
        Arguments:
            adapter: An [Adapter](../../adapters/index.md) object
            logger:
            log_freq: The number of iterations between logging
            with_pbars: If ```True```, progress bars are shown during
                each epoch.
        """
        self.adapter = adapter
        self.validator = validator
        self.stat_getter = stat_getter
        self.saver = saver
        self.logger = c_f.default(logger, IgniteEmptyLogger, {})
        self.val_data_hook = val_data_hook
        self.log_freq = log_freq
        self.with_pbars = with_pbars
        self.device = c_f.default(device, idist.device, {})
        self.trainer_init()
        self.collector_init()
        self.dist_init_done = False
        if device is None and auto_dist:
            self.dist_init()
        self.temp_events = []

    def training_step(self, engine, batch):
        batch = c_f.batch_to_device(batch, self.device)
        return self.adapter.training_step(batch)

    def before_training_starts(self, engine):
        self.adapter.before_training_starts(self)

    def trainer_init(self):
        self.trainer = Engine(self.training_step)
        self.trainer.state.adapter = self.adapter
        i_g.register(self.trainer, Events.STARTED, self.before_training_starts)
        i_g.register(
            self.trainer, Events.EPOCH_STARTED, self.set_to_train(self.adapter.models)
        )
        iteration_complete = [
            i_g.step_lr_schedulers(self.adapter.lr_schedulers, "per_step"),
            TerminateOnNan(),
        ]
        pbars = i_g.set_loggers_and_pbars(self, ["trainer"])
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

    def collector_init(self):
        self.collector = Engine(self.get_collector_step(self.adapter.inference))
        i_g.set_loggers_and_pbars(self, ["collector"])
        i_g.register(
            self.collector,
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
        datasets=None,
        dataloader_creator=None,
        dataloaders=None,
        validation_interval=1,
        patience=10,
        resume=None,
        check_initial_score=False,
        **trainer_kwargs,
    ):
        if dataloaders is None:
            dataloader_creator = c_f.default(dataloader_creator, DataloaderCreator())
            dataloaders = dataloader_creator(**datasets)

        self.remove_temp_events()

        if self.validator:
            max_epochs = trainer_kwargs.get("max_epochs", 1)
            self.add_validation_runner(
                dataloaders,
                validation_interval,
                max_epochs,
                check_initial_score,
            )

            self.add_temp_event_handler(
                Events.EPOCH_STARTED, i_g.early_stopper(patience, self.validator)
            )

        if self.saver:
            if not self.validator:
                self.add_temp_event_handler(
                    Events.EPOCH_COMPLETED,
                    i_g.save_adapter_without_validator(self.saver, self.adapter),
                )
            self.add_temp_event_handler(Events.EPOCH_COMPLETED, self.saver.save_ignite)

        if resume is not None:
            if resume != "latest":
                raise ValueError("Only 'latest' resume is currently supported")
            if not self.saver:
                raise ValueError("To resume, a Saver must be provided")
            self.saver.load_all(
                adapter=self.adapter,
                validator=self.validator,
                stat_getter=self.stat_getter,
                framework=self,
                suffix=resume,
            )
            i_g.resume_checks(self.validator, self.stat_getter, self)

        if not i_g.is_done(self.trainer, **trainer_kwargs):
            self.trainer.run(dataloaders["train"], **trainer_kwargs)

        if self.validator:
            return self.validator.best_score, self.validator.best_epoch

        return None, None

    def add_validation_runner(
        self,
        dataloaders,
        validation_interval,
        max_epochs,
        check_initial_score,
    ):
        validation_condition = Events.EPOCH_COMPLETED(every=validation_interval)
        if max_epochs % validation_interval != 0:
            validation_condition |= Events.EPOCH_COMPLETED(once=max_epochs)
        if check_initial_score:
            validation_condition |= Events.STARTED
        self.add_temp_event_handler(
            validation_condition,
            i_g.get_validation_runner(
                self,
                dataloaders,
                self.validator,
                self.stat_getter,
                self.saver,
                self.logger,
                self.val_data_hook,
            ),
        )

    def evaluate_best_model(self, datasets, validator, saver, dataloader_creator=None):
        c_f.LOGGER.info("***EVALUATING BEST MODEL***")
        dataloader_creator = c_f.default(dataloader_creator, DataloaderCreator())
        dataloaders = dataloader_creator(**datasets)
        saver.load_adapter(self.adapter, "best")
        collected_data = i_g.collect_from_dataloaders(
            self, dataloaders, validator.required_data
        )
        return i_g.get_validation_score(collected_data, validator)

    def get_collector_step(self, inference):
        def collector_step(engine, batch):
            batch = c_f.batch_to_device(batch, self.device)
            return f_utils.collector_step(inference, batch, f_utils.create_output_dict)

        return collector_step

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

    def add_temp_event_handler(self, event, handler):
        self.trainer.add_event_handler(event, handler)
        self.temp_events.append((handler, event))

    def remove_temp_events(self):
        for h, e in self.temp_events:
            self.trainer.remove_event_handler(h, e)
        self.temp_events = []

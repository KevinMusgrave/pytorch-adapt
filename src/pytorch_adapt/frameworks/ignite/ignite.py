from typing import Tuple, Union

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan

from ...datasets import DataloaderCreator
from ...utils import common_functions as c_f
from ...validators import utils as val_utils
from .. import utils as f_utils
from . import checkpoint_utils
from . import utils as i_g


class Ignite:
    """
    Wraps an [Adapter][pytorch_adapt.adapters.BaseAdapter] and takes
    care of validation, model saving, etc. by using
    the event handler system of PyTorch Ignite.
    """

    def __init__(
        self,
        adapter,
        validator=None,
        val_hooks=None,
        checkpoint_fn=None,
        logger=None,
        log_freq=50,
        with_pbars=True,
        device=None,
        auto_dist=True,
        val_output_dict_fn=None,
    ):
        """
        Arguments:
            adapter: An [```Adapter```][pytorch_adapt.adapters.BaseAdapter] object, which contains
                the training and inference steps.
            validator: An optional [```ScoreHistory```][pytorch_adapt.validators.ScoreHistory] object,
                which is used to determine the best checkpoint.
            val_hooks: A list of functions that are called during validation. Each function should
                be one of the following types:

                - [```Validator```][pytorch_adapt.validators.BaseValidator]
                - [```ScoreHistory```][pytorch_adapt.validators.ScoreHistory]
                - A callable that accepts the following arguments:
                    - ```epoch``` (an integer)
                    - ```**kwargs```, keyword arguments mapping from dataset split name
                        to dictionaries containing features, labels etc.

                You can give your custom callable a list attribute named ```required_data```.
                For example, setting ```self.required_data = ["src_train", "target_train"]```,
                will ensure that the validation runner gathers data for the
                ```"src_train"``` and ```"target_train"``` splits.


            checkpoint_fn: A [```CheckpointFnCreator```][pytorch_adapt.frameworks.ignite.CheckpointFnCreator] object.
            logger: An object with the following functions:

                - ```add_training(adapter)```
                - ```add_validation(data, epoch)```, where ```data``` is ```{validator_name: validator}```,
                    and ```validator_name``` is usually just ```"validator"```.
                - ```write(engine)```

            log_freq: The number of iterations between logging
            with_pbars: If ```True```, progress bars are shown during each epoch.
            device: If ```None```, then it defaults to [```idist.device()```](https://pytorch.org/ignite/v0.4.8/distributed.html#ignite.distributed.utils.device).
            auto_dist: If ```True``` and ```device == None``` then
                [```auto_model```](https://pytorch.org/ignite/v0.4.8/generated/ignite.distributed.auto.auto_model.html)
                and [```auto_optim```](https://pytorch.org/ignite/v0.4.8/generated/ignite.distributed.auto.auto_optim.html) are applied
                to the models and optimizers.
        """
        self.adapter = adapter
        self.validator = validator
        self.val_hooks = c_f.default(val_hooks, [])
        self.checkpoint_fn = checkpoint_fn
        self.logger = logger
        self.log_freq = log_freq
        self.with_pbars = with_pbars
        self.device = c_f.default(device, idist.device, {})
        self.val_output_dict_fn = c_f.default(
            val_output_dict_fn, f_utils.create_output_dict
        )
        self.trainer_init()
        self.collector_init()
        self.dist_init_done = False
        if device is None and auto_dist:
            self.dist_init()
        self.temp_events = []

    def training_step(self, engine, batch):
        batch = c_f.batch_to_device(batch, self.device)
        return self.adapter.training_step(batch)

    def trainer_init(self):
        self.trainer = Engine(self.training_step)
        i_g.register(self.trainer, Events.STARTED, *self.trainer_started_events())
        i_g.register(
            self.trainer, Events.EPOCH_STARTED, *self.trainer_epoch_started_events()
        )
        i_g.register(
            self.trainer,
            Events.ITERATION_COMPLETED,
            *self.trainer_iteration_complete_events(),
        )
        i_g.register(
            self.trainer,
            Events.ITERATION_COMPLETED(every=self.log_freq),
            *self.trainer_log_freq_events(),
        )
        i_g.register(
            self.trainer, Events.EPOCH_COMPLETED, *self.trainer_epoch_complete_events()
        )

    def trainer_started_events(self):
        def fn(engine):
            self.adapter.before_training_starts(self)

        return [fn]

    def trainer_epoch_started_events(self):
        return [self.set_to_train(self.adapter.models)]

    def trainer_iteration_complete_events(self):
        output = [
            i_g.step_lr_schedulers(self.adapter.lr_schedulers, "per_step"),
            TerminateOnNan(),
        ]
        pbars = i_g.set_loggers_and_pbars(self, ["trainer"])
        if self.with_pbars:
            output.append(i_g.pbar_print_losses(pbars["trainer"]))
        return output

    def trainer_log_freq_events(self):
        output = []
        if self.logger:
            output.append(self.logger.add_training(self.adapter))
        return output

    def trainer_epoch_complete_events(self):
        output = [
            i_g.step_lr_schedulers(self.adapter.lr_schedulers, "per_epoch"),
            i_g.zero_grad(self.adapter),
        ]
        if self.logger:
            output.append(self.logger.write)
        return output

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

    def get_all_outputs(self, dataloader, split_name):
        dataloaders = {split_name: dataloader}
        return i_g.collect_from_dataloaders(self.collector, dataloaders, [split_name])

    def run(
        self,
        datasets=None,
        dataloader_creator=None,
        dataloaders=None,
        val_interval=1,
        early_stopper_kwargs=None,
        resume=None,
        check_initial_score=False,
        **trainer_kwargs,
    ) -> Union[Tuple[float, int], Tuple[None, None]]:
        """
        Trains and optionally validates on the input datasets.

        Arguments:
            datasets: A dictionary mapping from dataset split names to datasets.
            dataloader_creator: Creates dataloaders using ```datasets```,
                when ```dataloaders == None```. Default value is
                [```DataloaderCreator()```][pytorch_adapt.datasets.DataloaderCreator].
            dataloaders: A dictionary mapping from dataset split names to
                dataloaders. If ```None```, then ```datasets``` must be provided.
            val_interval: ```self.validator``` and ```self.val_hooks``` will be called
                every ```val_interval``` epochs.
            early_stopper_kwargs: A dictionary of keyword arguments to be passed to
                [```EarlyStopping```](https://pytorch.org/ignite/v0.4.8/generated/ignite.handlers.early_stopping.EarlyStopping.html).
            resume: Resume training from a checkpoint. This should be either
                a string representing a file path, or an integer representing a global step.
            check_initial_score: If ```True```, then ```self.validator``` and ```self.val_hooks```
                will be called before training starts. For example, you might use this to check the performance of
                a pretrained model before finetuning.
            **trainer_kwargs: Keyword arguments that are passed to the PyTorch Ignite
                [```run()```](https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.run)
                function.
        Returns:
            A tuple of ```(best_score, best_epoch)``` or ```(None, None)```
            if no validator is used.
        """
        if dataloaders is None:
            dataloader_creator = c_f.default(dataloader_creator, DataloaderCreator())
            dataloaders = dataloader_creator(**datasets)

        self.remove_temp_events()
        max_epochs = trainer_kwargs.get("max_epochs", 1)
        condition = i_g.interval_condition(val_interval, max_epochs)

        if self.checkpoint_fn:
            self.add_checkpoint_fn(condition, dataloaders)
        elif self.validator or self.val_hooks:
            self.add_validation_runner(condition, dataloaders)

        if check_initial_score:
            self.add_validation_runner(Events.STARTED, dataloaders)

        if self.validator and early_stopper_kwargs:
            self.add_early_stopper(val_interval, **early_stopper_kwargs)

        if resume is not None:
            self.load_checkpoint(resume)

        if not i_g.is_done(self.trainer, max_epochs):
            self.trainer.run(dataloaders["train"], **trainer_kwargs)

        self.remove_temp_events()

        if self.validator:
            return self.validator.best_score, self.validator.best_epoch

        return None, None

    def get_validation_runner(
        self,
        dataloaders,
    ):
        return i_g.get_validation_runner(
            self.collector,
            dataloaders,
            self.validator,
            self.val_hooks,
            self.logger,
        )

    def add_validation_runner(self, condition, dataloaders):
        val_runner = self.get_validation_runner(dataloaders)
        self.add_temp_event_handler(condition, val_runner)

    def add_checkpoint_fn(self, condition, dataloaders):
        score_function = (
            self.get_validation_runner(dataloaders) if self.validator else None
        )
        self.add_temp_event_handler(
            condition,
            self.checkpoint_fn(
                adapter=self.adapter,
                validator=self.validator,
                val_hooks=self.val_hooks,
                score_function=score_function,
            ),
        )
        if not self.validator and self.val_hooks:
            self.add_validation_runner(condition, dataloaders)

    def add_early_stopper(self, val_interval, **kwargs):
        def score_fn(_):
            return self.validator.latest_score

        self.add_temp_event_handler(
            Events.EPOCH_COMPLETED(every=val_interval),
            i_g.early_stopper(**kwargs)(
                trainer=self.trainer,
                score_function=score_fn,
            ),
        )

    def load_checkpoint(self, resume):
        to_load = {
            "engine": self.trainer,
            "validator": self.validator,
            **checkpoint_utils.adapter_to_dict(self.adapter),
            **checkpoint_utils.val_hooks_to_dict(self.val_hooks),
        }
        if isinstance(resume, str):
            kwargs = {"checkpoint": resume}
        elif isinstance(resume, int):
            kwargs = {"global_step": resume}
        else:
            raise TypeError(
                "resume must be a string representing a file path, or an integer representing a global step"
            )
        self.checkpoint_fn.load_objects(to_load, **kwargs)
        i_g.resume_checks(self.trainer, self.validator)

    def evaluate_best_model(
        self, datasets, validator, dataloader_creator=None
    ) -> float:
        """
        Loads the best checkpoint and computes the score on the given datasets.
            This requires ```self.checkpoint_fn``` to be not ```None```.
        Arguments:
            datasets: A dictionary mapping from dataset split names to datasets.
            validator: A [Validator][pytorch_adapt.validators.BaseValidator]
                or [ScoreHistory][pytorch_adapt.validators.ScoreHistory] object,
                which will compute and return the score.
            dataloader_creator: If ```None```, it will default to
                [```DataloaderCreator()```][pytorch_adapt.datasets.DataloaderCreator].
        Returns:
            The validator's score for the best checkpoint.
        """
        c_f.LOGGER.info("***EVALUATING BEST MODEL***")
        dataloader_creator = c_f.default(dataloader_creator, DataloaderCreator, {})
        dataloaders = dataloader_creator(**datasets)
        self.checkpoint_fn.load_best_checkpoint({"models": self.adapter.models})
        collected_data = i_g.collect_from_dataloaders(
            self.collector, dataloaders, validator.required_data
        )
        return val_utils.call_val_hook(validator, collected_data)

    def get_collector_step(self, inference):
        def collector_step(engine, batch):
            batch = c_f.batch_to_device(batch, self.device)
            return f_utils.collector_step(inference, batch, self.val_output_dict_fn)

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
        removable = self.trainer.add_event_handler(event, handler)
        self.temp_events.append(removable)

    def remove_temp_events(self):
        for h in self.temp_events:
            h.remove()
        self.temp_events = []

import logging

import ignite.distributed as idist
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from ignite.utils import setup_logger

from ...utils import common_functions as c_f
from ...utils import exceptions
from ...validators import utils as val_utils
from ...weighters import get_multiple_loss_totals
from .dictionary_accumulator import DictionaryAccumulator


def get_validation_runner(
    collector,
    dataloaders,
    validator,
    val_hooks,
    logger,
):
    required_data = []
    for v in [validator, *val_hooks]:
        if v and hasattr(v, "required_data"):
            required_data = list(set(required_data + v.required_data))

    def run_validation(engine):
        epoch = engine.state.epoch
        collected_data = collect_from_dataloaders(collector, dataloaders, required_data)
        score = None
        if validator:
            score = val_utils.call_val_hook(validator, collected_data, epoch)
        for hook in val_hooks:
            val_utils.call_val_hook(hook, collected_data, epoch)
        if logger:
            logger.add_validation({"validator": validator}, epoch)
            logger.write(engine)
        return score

    return run_validation


def collect_from_dataloaders(collector, dataloaders, required_data):
    collected_data = {}

    for k in required_data:
        c_f.LOGGER.info(f"Collecting {k}")
        curr_dataloader = dataloaders[k]
        c_f.val_dataloader_checks(curr_dataloader)
        curr_dataset = curr_dataloader.dataset
        iterable = curr_dataloader.__iter__()
        curr_collected = accumulate_collector_output(collector, iterable, k)
        c_f.val_collected_data_checks(curr_collected, curr_dataset)
        collected_data[k] = curr_collected
        del iterable

    return collected_data


def step_lr_schedulers(lr_schedulers, scheduler_type):
    def handler(engine):
        lr_schedulers.step(scheduler_type)

    return handler


def auto_model(*args, **kwargs):
    def handler(model):
        return idist.auto_model(model, *args, **kwargs)

    return handler


def pbar_print_losses(pbar):
    def handler(engine):
        losses = get_multiple_loss_totals(engine.state.output)
        losses = [f"{k}={v:6.4f}" for k, v in losses.items()]
        pbar.pbar.unit = ", ".join(losses)

    return handler


def get_default_pbar():
    bar_format = "{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}|{unit} [{elapsed}<{remaining}]"
    return ProgressBar(persist=True, bar_format=bar_format)


def attach_pbar(engine, *_):
    pbar = get_default_pbar()
    pbar.attach(engine)
    return pbar


def accumulate_collector_output(collector, iterable, output_name):
    accumulator = DictionaryAccumulator()
    accumulator.attach(collector, output_name)
    collector.run(iterable)
    accumulator.detach(collector)
    output = collector.state.metrics[output_name]
    collector.state.output = {}
    collector.state.metrics = {}
    return output


def set_engine_logger(engine, name, level=logging.CRITICAL):
    engine.logger = setup_logger(name=name, level=level)
    return engine.logger


def register(engine, event, *args):
    for h in args:
        engine.add_event_handler(event, h)


def do_for_all_engines(cls, function, keys):
    output = {}
    for name in keys:
        output[name] = function(getattr(cls, name), name)
    return output


def set_loggers_and_pbars(cls, keys):
    do_for_all_engines(cls, set_engine_logger, keys)
    if cls.with_pbars:
        return do_for_all_engines(cls, attach_pbar, keys)


def resume_checks(trainer, validator):
    last_trainer_epoch = trainer.state.epoch
    if not validator:
        return
    last_validator_epoch = validator.epochs[-1]
    if last_trainer_epoch != last_validator_epoch:
        raise exceptions.ResumeCheckError(
            f"Last trainer epoch ({last_trainer_epoch}) does not equal last validator epoch ({last_validator_epoch})"
        )


def is_done(trainer, max_epochs):
    return trainer.state.epoch >= max_epochs


def zero_grad(adapter):
    def handler(engine):
        c_f.LOGGER.debug("zeroing grads")
        adapter.models.zero_grad()
        adapter.optimizers.zero_grad()

    return handler


def interval_condition(interval, max_epochs):
    condition = Events.EPOCH_COMPLETED(every=interval)
    if max_epochs % interval != 0:
        condition |= Events.EPOCH_COMPLETED(once=max_epochs)
    return condition


def early_stopper(**kwargs):
    def fn(trainer, score_function):
        return EarlyStopping(trainer=trainer, score_function=score_function, **kwargs)

    return fn

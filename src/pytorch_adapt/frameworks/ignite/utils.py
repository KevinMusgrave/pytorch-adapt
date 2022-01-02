import logging

import ignite.distributed as idist
from ignite.contrib.handlers import ProgressBar
from ignite.utils import setup_logger

from ...utils import common_functions as c_f
from ...utils import exceptions
from ...weighters import get_multiple_loss_totals
from .dictionary_accumulator import DictionaryAccumulator


def get_validation_runner(
    cls,
    dataloaders,
    validator,
    stat_getter,
    saver,
    logger,
    val_data_hook,
):
    required_data = validator.required_data
    if stat_getter:
        required_data = list(set(required_data + stat_getter.required_data))

    def run_validation(engine):

        epoch = engine.state.epoch

        collected_data = collect_from_dataloaders(cls, dataloaders, required_data)
        get_validation_score(collected_data, validator, epoch)

        if saver:
            saver.save_validator(validator)
            saver.save_adapter(cls.adapter, epoch, validator.best_epoch)
        if logger:
            logger.add_validation({"validator": validator}, epoch)
        log_str = f"VALIDATION SCORES:\n{validator}\n"

        if stat_getter:
            get_validation_score(collected_data, stat_getter, epoch)
            if saver:
                saver.save_stat_getter(stat_getter)
            if logger:
                logger.add_validation({"stat_getter": stat_getter}, epoch)
            log_str += f"OTHER STATS:\n{stat_getter}\n"

        c_f.LOGGER.info(log_str)
        if logger:
            logger.write(engine)

        if val_data_hook:
            val_data_hook(engine, collected_data)

    return run_validation


def get_validation_score(collected_data, validator, epoch=None):
    kwargs = c_f.filter_kwargs(collected_data, validator.required_data)
    if epoch is None:
        return validator.score(**kwargs)
    return validator.score(epoch, **kwargs)


def collect_from_dataloaders(cls, dataloaders, required_data):
    collected_data = {}

    for k in required_data:
        c_f.LOGGER.info(f"Collecting {k}")
        curr_dataloader = dataloaders[k]
        c_f.val_dataloader_checks(curr_dataloader)
        curr_dataset = curr_dataloader.dataset
        iterable = curr_dataloader.__iter__()
        curr_collected = accumulate_collector_output(cls.collector, iterable, k)
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


def save_adapter_without_validator(saver, adapter):
    def handler(engine):
        saver.save_adapter(adapter, engine.state.epoch, None)

    return handler


def early_stopper(patience, validator):
    def handler(engine):
        if engine.state.epoch > 2:
            # this runs at the beginning of a new epoch
            # so engine.state.epoch has already incremented
            # it's also 1-indexed
            epochs_since_best_epoch = engine.state.epoch - 1
            if validator.best_epoch is not None:
                epochs_since_best_epoch -= validator.best_epoch
            c_f.LOGGER.info(f"epochs_since_best_epoch = {epochs_since_best_epoch}")
            if epochs_since_best_epoch > patience:
                c_f.LOGGER.info("***Performance has plateaued. Exiting.***")
                engine.terminate()

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
    return collector.state.metrics[output_name]


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


def resume_checks(validator, stat_getter, framework):
    last_trainer_epoch = framework.trainer.state.epoch
    for name, v in {"validator": validator, "stat_getter": stat_getter}.items():
        if not v:
            continue
        last_validator_epoch = v.epochs[-1]
        if last_trainer_epoch != last_validator_epoch:
            raise exceptions.ResumeCheckError(
                f"Last trainer epoch ({last_trainer_epoch}) does not equal last {name} epoch ({last_validator_epoch})"
            )


def is_done(trainer, max_epochs=None, **kwargs):
    try:
        return trainer.state.epoch >= max_epochs
    except TypeError:
        return False


def zero_grad(adapter):
    def handler(engine):
        c_f.LOGGER.debug("zeroing grads")
        adapter.models.zero_grad()
        adapter.optimizers.zero_grad()

    return handler

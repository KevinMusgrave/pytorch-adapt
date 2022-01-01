import errno
import glob
import itertools
import json
import logging
import os
import re
import shutil
import tarfile
import zipfile

import numpy as np
import torch
import tqdm
from pytorch_metric_learning.utils import common_functions as pml_cf

LOGGER_NAME = "pytorch-adapt"
LOGGER = logging.getLogger(LOGGER_NAME)


def set_logger_name(name):
    global LOGGER_NAME
    global LOGGER
    LOGGER_NAME = name
    LOGGER = logging.getLogger(LOGGER_NAME)


def has_no_parameters(model):
    return len(list(model.parameters())) == 0


def is_optimizer(optimizer):
    return isinstance(optimizer, torch.optim.Optimizer)


def is_none(x):
    return x is None


def default(x, default_x, kwargs=None, condition=None):
    if condition is None:
        condition = is_none
    if kwargs is not None:
        if isinstance(kwargs, dict):
            return default_x(**kwargs) if condition(x) else x
        if isinstance(kwargs, list):
            return default_x(*kwargs) if condition(x) else x
    return default_x if condition(x) else x


def many_default(x, default_x, kwargs=None, condition=None):
    return [default(x[i], default_x[i], kwargs, condition) for i in range(len(x))]


def dict_pop_lazy(x, key, *args, **kwargs):
    y = x.pop(key, None)
    return default(y, *args, **kwargs)


def add_if_new(d, key, x, kwargs, model_name, in_keys, other_args=None):
    # if key is list then assume model returns multiple args
    if not is_list_or_tuple(key) or not is_list_or_tuple(x):
        raise TypeError("key and x must both be a list or tuple")
    condition = is_none
    if any(condition(y) for y in x):
        model = kwargs[model_name]
        input_vals = [kwargs[k] for k in in_keys]
        if other_args is not None:
            input_vals += other_args
        new_x = default(None, model, input_vals, is_none)
        if len(x) > 1:
            if not is_list_or_tuple(new_x) or len(new_x) != len(x):
                raise TypeError(
                    "if input x and key are lists, then output of model must be a list of the same length"
                )
            for i in range(len(x)):
                if condition(x[i]):
                    d[key[i]] = new_x[i]
        else:
            d[key[0]] = new_x


def class_default(cls, x, default):
    return x(cls) if x else default


def list_diff(x, y):
    return list(set(x) - set(y))


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def batch_to_device(batch, device):
    if is_list_or_tuple(batch):
        return [pml_cf.to_device(x, device=device) for x in batch]
    if isinstance(batch, dict):
        return {k: pml_cf.to_device(x, device=device) for k, x in batch.items()}
    return pml_cf.to_device(batch, device=device)


def first_key(in_dict):
    return list(in_dict.keys())[0]


def first_val(in_dict):
    return in_dict[first_key(in_dict)]


def kronecker_product(x, y):
    batch_size = x.shape[0]
    tensor_prod = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))
    return tensor_prod.view(batch_size, -1)


def makedir_if_not_there(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def filter_kwargs(kwargs, keep):
    return {x: kwargs[x] for x in keep}


def val_dataloader_checks(dataloader):
    if dataloader.drop_last:
        raise ValueError("drop_last should be False when doing validation")
    if not isinstance(dataloader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError("shuffle should be False when doing validation")


def val_collected_data_checks(collected, dataset):
    for k, v in collected.items():
        if len(v) != len(dataset):
            raise ValueError(
                f"The length of {k} should be equal to the length of the validation dataset"
            )


def add_prefix(x, prefix=""):
    prefix = f"{prefix}_" if prefix != "" else prefix
    return f"{prefix}{x}"


def add_suffix(x, suffix=""):
    suffix = f"_{suffix}" if suffix != "" else suffix
    return f"{x}{suffix}"


def class_as_prefix(obj, x, prefix="", suffix=""):
    x = add_prefix(x, cls_name(obj))
    x = add_prefix(x, prefix)
    x = add_suffix(x, suffix)
    return x


def full_path(folder, filename):
    return os.path.join(folder, filename)


def has_state_dict(x):
    return hasattr(x, "state_dict")


def copy_file(src, dst):
    shutil.copyfile(src, dst)


def save_torch_module(x, folder, filename):
    if isinstance(x, torch.nn.DataParallel):
        x = x.module
    makedir_if_not_there(folder)
    filename = full_path(folder, filename)
    LOGGER.debug(f"Saving {filename}")
    pml_cf.save_model(x, filename)


def load_torch_module(x, folder, filename):
    if isinstance(x, torch.nn.DataParallel):
        x = x.module
    filename = full_path(folder, filename)
    LOGGER.debug(f"Loading {filename}")
    pml_cf.load_model(x, filename, torch.device("cuda"))


def save_json(x, folder, filename):
    makedir_if_not_there(folder)
    filename = full_path(folder, filename)
    with open(filename, "w") as fp:
        LOGGER.debug(f"Saving {filename}")
        json.dump(x, fp)


def load_json(folder, filename):
    filename = full_path(folder, filename)
    with open(filename, "r") as fp:
        LOGGER.debug(f"Loading {filename}")
        data = json.load(fp)
    return data


def save_npy(x, folder, filename):
    makedir_if_not_there(folder)
    filename = full_path(folder, filename)
    LOGGER.debug(f"Saving {filename}")
    np.save(filename, x, allow_pickle=False)


def load_npy(folder, filename):
    filename = full_path(folder, filename)
    LOGGER.debug(f"Loading {filename}")
    return np.load(filename, allow_pickle=False)


def enumerate_to_dict(x):
    if isinstance(x, list):
        return {str(i): v for i, v in enumerate(x)}
    elif isinstance(x, dict):
        return x
    else:
        raise TypeError("input must be a list or dict")


def delete_all_but(folder, basename, extension, to_keep):
    basenames_to_keep = []
    for prefix, suffix in to_keep:
        new_basename = add_suffix(add_prefix(basename, prefix), suffix)
        basenames_to_keep.append(new_basename)
    files_to_keep = [full_path(folder, f"{x}{extension}") for x in basenames_to_keep]
    all_matching_files = glob.glob(full_path(folder, f"*{basename}*{extension}"))

    for f in all_matching_files:
        if f not in files_to_keep:
            LOGGER.debug(f"Deleting {f}")
            os.remove(f)


def state_dicts_are_equal(x, y, rtol=None):
    if x.keys() != y.keys():
        return False
    for k, v1 in x.items():
        v2 = y[k]
        if type(v1) != type(v2):
            return False
        if torch.is_tensor(v1):
            if rtol is None and not torch.equal(v1, v2):
                return False
            elif rtol and not torch.allclose(v1, v2, rtol=rtol):
                return False
        elif isinstance(v1, dict) and not state_dicts_are_equal(v1, v2):
            return False
    return True


def set_layers_mode(mode, layer_types=None, layer_names=None):
    if layer_types is None and layer_names is None:
        raise ValueError("layer_types and layer_names cannot both be None")
    if mode not in ["train", "eval"]:
        raise ValueError("mode must be either 'train' or 'eval'")

    def helper(m, mode):
        m.eval() if mode == "eval" else m.train()

    def set_to_mode(m):
        if layer_names is not None:
            classname = cls_name(m)
            if any(L in classname for L in layer_names):
                helper(m, mode)
        if layer_types is not None:
            if isinstance(m, layer_types):
                helper(m, mode)

    return set_to_mode


def batchnorm_types():
    return (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def cls_name(x):
    return x.__class__.__name__


def assert_keys_are_present_cls(cls, attr_name, x):
    assert_keys_are_present(getattr(cls, attr_name), attr_name, x)


def assert_keys_are_present(attr, attr_name, x):
    if any(k not in x for k in attr.keys()):
        raise KeyError(f"{attr_name} ({attr}) has keys that are not in input ({x}).")


def get_or_pop(x, k, pop):
    return x.pop(k) if pop else x[k]


def extract(x, keys, pop=False):
    if isinstance(x, list):
        output = []
        for k in keys:
            success = False
            for y in x:
                try:
                    output.append(get_or_pop(y, k, pop))
                    success = True
                    break
                except KeyError:
                    continue
            if not success:
                raise KeyError(f"{k} not found")
        return output
    else:
        return [get_or_pop(x, k, pop) for k in keys]


def filter(x, regex_str, return_order=None):
    r = re.compile(regex_str)
    x = [y for y in x if r.search(y)]
    if return_order:
        output = []
        for s in return_order:
            r = re.compile(s)
            output.extend([y for y in x if r.search(y)])
        return output
    return x


def to_dict(x, keys):
    return {k: x[i] for i, k in enumerate(keys)}


def concat_tensor_dict(x, keys, dim):
    return torch.cat(extract(x, keys), dim=dim)


def zero_loss():
    return torch.tensor(0, dtype=torch.float, requires_grad=True)


def zero_back_step(loss, optimizers, custom_backward=None):
    for x in optimizers:
        x.zero_grad()
    if loss.grad_fn is not None:
        if custom_backward:
            custom_backward(loss)
        else:
            loss.backward()
        for x in optimizers:
            x.step()


def len_one_tensor(x):
    return x.dim() == 0 or len(x) == 1


# https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/8
def reinit_layer(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


# https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/8
def reinit(model):
    model.apply(reinit_layer)
    return model


def map_keys(x, key_map, must_be_subset=True):
    if must_be_subset and not set(key_map.keys()).issubset(set(x.keys())):
        raise KeyError("key_map keys must be a subset of the input")
    output = {}
    for k, v in x.items():
        new_k = key_map.get(k, k)
        if new_k != k or new_k not in output:
            output[new_k] = v
    return output


# https://stackoverflow.com/a/6117124
def map_keys_substrings(x, key_map):
    rep = dict((re.escape(k), v) for k, v in key_map.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], x)


def join_lists(x):
    return list(itertools.chain(*x))


def attrs_of_type(cls, obj):
    attrs = [a for a in dir(cls) if not a.startswith("__")]
    attrs = {h: getattr(cls, h) for h in attrs}
    return {k: v for k, v in attrs.items() if isinstance(v, obj)}


def append_error_message(e, msg):
    if len(e.args) >= 1:
        e.args = (e.args[0] + msg,) + e.args[1:]


def requires_grad(x, does=True):
    if isinstance(x, (list, tuple, set)):
        return all(requires_grad(y, does) for y in x)
    if isinstance(x, (dict)):
        return all(requires_grad(y, does) for y in x.values())
    x = x.requires_grad
    return x if does else not x


def to_set(x):
    if isinstance(x, list):
        if all(isinstance(y, dict) for y in x):
            return set().union(*x)
        return set(x)
    return set(x)


def check_domain(cls, domain, keep_len=False):
    if domain is not None:
        if len(torch.unique(domain)) > 1:
            raise ValueError(
                f"{cls_name(cls)} inference only supports one domain per batch"
            )
        if not keep_len and not len_one_tensor(domain):
            domain = domain[0]
    return domain


def extra_repr(cls, attr_names, delimiter="\n"):
    return delimiter.join([f"{a}={str(getattr(cls, a))}" for a in attr_names])


# copied from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
def add_indent(s_, num_spaces=2, not_first=False):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    if not_first:
        first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    if not_first:
        s = first + "\n" + s
    return s


# copied from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
def nice_repr(cls, extra_repr, children):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, value in children.items():
        mod_str = repr(value)
        mod_str = add_indent(mod_str, 2, not_first=True)
        child_lines.append("(" + key + "): " + mod_str)
    lines = extra_lines + child_lines

    main_str = cls_name(cls) + "("
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += ")"
    return main_str


def dicts_are_overlapping(x, y, return_overlap=False):
    overlap = x.keys() & y.keys()
    is_overlap = len(overlap) > 0
    if return_overlap:
        return is_overlap, overlap
    return is_overlap


def assert_dicts_are_disjoint(*x):
    output, total_len = {}, 0
    for y in x:
        output.update(y)
        total_len += len(y)
    if len(output) != total_len:
        raise KeyError(f"dicts have overlapping keys")
    return output


def extract_progress(compressed_obj):
    LOGGER.info("Extracting dataset")
    if isinstance(compressed_obj, tarfile.TarFile):
        iterable = compressed_obj
        length = len(compressed_obj.getmembers())
    elif isinstance(compressed_obj, zipfile.ZipFile):
        iterable = compressed_obj.namelist()
        length = len(iterable)
    for member in tqdm.tqdm(iterable, total=length):
        yield member

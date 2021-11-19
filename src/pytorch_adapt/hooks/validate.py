import copy
from collections import defaultdict
from typing import Dict

from ..utils import common_functions as c_f
from .base import BaseHook, check_keys_are_present
from .features import BaseFeaturesHook
from .utils import ChainHook, ParallelHook, RepeatHook


def update_model_counts(hook, available_keys, model_counts):
    # get model counts
    # will be correct only for non-conditional hooks
    if isinstance(hook, BaseFeaturesHook):
        needs_compute = set(hook.out_keys) - available_keys
        needs_compute_by_domain = defaultdict(list)
        for nc in needs_compute:
            needs_compute_by_domain[nc.split("_")[0]].append(nc)
        for domain, ncs in needs_compute_by_domain.items():
            if not hook.detach[domain]:
                model_counts[hook.model_name] += 1
            else:
                for nc in ncs:
                    detachable_regex = hook.detachable_regex[nc]
                    if not any(detachable_regex.search(ak) for ak in available_keys):
                        model_counts[hook.model_name] += 1
                        break


def validate_hook(
    hook, available_keys=None, depth=0, model_counts=None
) -> Dict[str, int]:
    """
    Arguments:
        hook: the hook to validate
        available_keys: a list of keys that the context
            will start with.
    Returns:
        A dictionary with each model's ```forward``` call count.
    """
    c_f.LOGGER.debug(f"VALIDATE: {'  '*depth}{c_f.cls_name(hook)}")
    available_keys = c_f.default(available_keys, [])
    model_counts = c_f.default(model_counts, defaultdict(int))

    if isinstance(available_keys, list):
        available_keys = set(available_keys)

    if isinstance(hook, ChainHook):
        hooks = hook.hooks
        for i in range(0, len(hooks)):
            validate_hook(hooks[i], available_keys, depth + 1, model_counts)

    elif isinstance(hook, ParallelHook):
        hooks = hook.hooks
        for i in range(0, len(hooks)):
            curr_available_keys = copy.deepcopy(available_keys)
            validate_hook(hooks[i], curr_available_keys, depth + 1, model_counts)

    elif isinstance(hook, RepeatHook):
        for _ in range(hook.n):
            curr_available_keys = copy.deepcopy(available_keys)
            validate_hook(hook.hook, curr_available_keys, depth + 1, model_counts)

    else:
        check_keys_are_present(
            hook, hook.in_keys, list(available_keys), "in_keys", "available_keys"
        )
        check_keys_are_present(
            hook,
            list(hook.key_map.keys()),
            list(available_keys),
            "key_map",
            "available_keys",
        )
        all_hooks = c_f.attrs_of_type(hook, BaseHook)
        for h in all_hooks.values():
            validate_hook(h, available_keys, depth + 1, model_counts)
        update_model_counts(hook, available_keys, model_counts)
        available_keys.update(set(hook.out_keys))

    return model_counts

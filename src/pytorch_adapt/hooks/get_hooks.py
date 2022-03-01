from ..utils import common_functions as c_f
from .base import BaseHook
from .utils import ChainHook, ParallelHook


def get_child_hooks(hook):
    output = c_f.attrs_of_type(hook, BaseHook, return_as_list=True)
    if isinstance(hook, ChainHook):
        output.extend([*hook.hooks, *hook.conditions, *hook.alts])
    if isinstance(hook, ParallelHook):
        output.extend(hook.hooks)
    return output


def get_hooks(hook, query):
    output = []
    if isinstance(hook, query):
        output.append(hook)
    for child in get_child_hooks(hook):
        output.extend(get_hooks(child, query))
    return output

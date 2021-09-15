from typing import Callable, List, Union

from ..utils import common_functions as c_f
from .base import BaseConditionHook, BaseHook, BaseWrapperHook


class EmptyHook(BaseHook):
    """Returns two empty dictionaries."""

    def call(self, losses, inputs):
        """"""
        return {}, {}

    def _loss_keys(self):
        """"""
        return []

    def _out_keys(self):
        """"""
        return []


class ZeroLossHook(BaseHook):
    """
    Returns only 0 losses and ```None``` outputs.
    """

    def __init__(self, loss_names: List[str], out_names: List[str], **kwargs):
        """
        Arguments:
            loss_names: The keys of the loss dictionary
                which will have ```tensor(0.)``` as its values.
            out_names: The keys of the output dictionary
                which will have ```None``` as its values.
        """
        super().__init__(**kwargs)
        self.loss_names = loss_names
        self.out_names = out_names

    def call(self, losses, inputs):
        """"""
        out_keys = set(self.out_names) - inputs.keys()
        return {k: c_f.zero_loss() for k in self.loss_names}, {
            k: None for k in out_keys
        }

    def _loss_keys(self):
        """"""
        return self.loss_names

    def _out_keys(self):
        """"""
        return self.out_names


class ChainHook(BaseHook):
    """
    Calls multiple hooks sequentially.
    The Nth hook receives the context accumulated through hooks 0 to N-1.
    """

    def __init__(
        self,
        *hooks: BaseHook,
        conditions: List[BaseConditionHook] = None,
        alts: List[BaseHook] = None,
        overwrite: Union[bool, List[int]] = False,
        **kwargs,
    ):
        """
        Arguments:
            hooks: a sequence of hooks that will be called sequentially.
            conditions: an optional list of condition hooks.
                If conditions[i] returns False, then alts[i] is called. Otherwise hooks[i] is called.
            alts: an optional list of hooks that will be executed
                when the corresponding condition hook returns False
            overwrite: If True, then hooks will be allowed to overwrite keys in the context.
                If a list of integers, then the hooks at the specified indices
                will be allowed to overwrite keys in the context.

        """

        super().__init__(**kwargs)
        self.hooks = hooks
        self.conditions = c_f.default(
            conditions, [TrueHook() for _ in range(len(hooks))]
        )
        self.alts = c_f.default(
            alts, [ZeroLossHook(h.loss_keys, h.out_keys) for h in self.hooks]
        )
        self.check_alt_keys_match_hook_keys()
        if not isinstance(overwrite, (list, bool)):
            raise TypeError("overwrite must be a list or bool")
        self.overwrite = overwrite
        self.in_keys = self.hooks[0].in_keys

    def call(self, losses, inputs):
        """"""
        out_losses, outputs = {}, {}
        all_losses, all_inputs = losses, inputs
        prev_losses, prev_outputs = {}, {}
        for i, h in enumerate(self.hooks):
            self.check_overwrite(i, all_losses, prev_losses, False)
            self.check_overwrite(i, all_inputs, prev_outputs, self.overwrite)
            all_losses = {**all_losses, **prev_losses}
            all_inputs = {**all_inputs, **prev_outputs}
            if self.conditions[i](all_losses, all_inputs):
                x = h(all_losses, all_inputs)
            else:
                x = self.alts[i](all_losses, all_inputs)
            prev_losses, prev_outputs = x
            out_losses.update(prev_losses)
            outputs.update(prev_outputs)
        return out_losses, outputs

    def check_overlap(self, x, y, names):
        is_overlap, overlap = c_f.dicts_are_overlapping(x, y, return_overlap=True)
        if is_overlap:
            raise KeyError(
                f"overwrite is false, but {names[0]} and {names[1]} have overlapping keys: {overlap}"
            )

    def check_overwrite(self, i, kwargs, prev_outputs, overwrite):
        if not overwrite or (isinstance(overwrite, list) and i not in overwrite):
            self.check_overlap(kwargs, prev_outputs, ["kwargs", "prev_outputs"])

    def _loss_keys(self):
        """"""
        return c_f.join_lists([h.loss_keys for h in self.hooks])

    def _out_keys(self):
        """"""
        return c_f.join_lists([h.out_keys for h in self.hooks])

    @property
    def last_hook_out_keys(self):
        return self.hooks[-1].out_keys

    def check_alt_keys_match_hook_keys(self):
        for i in range(len(self.hooks)):
            h = self.hooks[i]
            a = self.alts[i]
            if (sorted(h.loss_keys) != sorted(a.loss_keys)) or (
                sorted(h.out_keys) != sorted(a.out_keys)
            ):
                raise ValueError(
                    "alt loss/out keys must be equal to hook loss/out keys"
                )

    def children_repr(self):
        x = super().children_repr()
        x["hooks"] = self.hooks
        if any(not isinstance(c, TrueHook) for c in self.conditions):
            x.update({"conditions": self.conditions, "alts": self.alts})
        return x


# outputs are not shared among parallel streams
# they are returned at the very end
class ParallelHook(BaseHook):
    """
    Calls multiple hooks while keeping contexts separate.
    The Nth hook receives the same context as hooks 0 to N-1.
    All the output contexts are merged at the end.
    """

    def __init__(self, *hooks: BaseHook, **kwargs):
        """
        Arguments:
            hooks: a sequence of hooks that will be called sequentially,
                with each hook receiving the same initial context.
        """
        super().__init__(**kwargs)
        self.hooks = hooks
        self.in_keys = c_f.join_lists([h.in_keys for h in self.hooks])

    def call(self, losses, inputs):
        """"""
        out_losses, outputs = {}, {}
        for h in self.hooks:
            x = h(losses, inputs)
            out_losses.update(x[0])
            outputs.update(x[1])

        return out_losses, outputs

    def children_repr(self):
        x = super().children_repr()
        x.update({"hooks": self.hooks})
        return x

    def _loss_keys(self):
        """"""
        return c_f.join_lists([h.loss_keys for h in self.hooks])

    def _out_keys(self):
        """"""
        return c_f.join_lists([h.out_keys for h in self.hooks])


class OnlyNewOutputsHook(BaseWrapperHook):
    """
    Returns only outputs that are not present in the input context.
    You should use this if you want to change the value of
    a key passed to self.hook, but not propagate that change
    to the outside.
    """

    def __init__(self, hook: BaseHook, **kwargs):
        """
        Arguments:
            hook: The hook inside which changes to the context will be allowed.
        """
        super().__init__(**kwargs)
        self.hook = hook

    def call(self, losses, inputs):
        """"""
        losses, outputs = self.hook(losses, inputs)
        outputs = {k: outputs[k] for k in (outputs.keys() - inputs.keys())}
        c_f.assert_dicts_are_disjoint(inputs, outputs)
        return losses, outputs


class ApplyFnHook(BaseHook):
    """
    Applies a function to specific values of the context.
    """

    def __init__(
        self, fn: Callable, apply_to: List[str], is_loss: bool = False, **kwargs
    ):
        """
        Arguments:
            fn: The function that will be applied to the inputs.
            apply_to: fn will be applied to ```inputs[k]``` for k in apply_to
            is_loss: If False, then the returned loss dictionary will be empty.
                Otherwise, the returned output dictionary will be empty.
        """
        super().__init__(**kwargs)
        self.fn = fn
        self.apply_to = apply_to
        self.is_loss = is_loss

    def call(self, losses, inputs):
        """"""
        x = c_f.extract(inputs, self.apply_to)
        outputs = {k: self.fn(v) for k, v in zip(self.apply_to, x)}
        if self.is_loss:
            return outputs, {}
        return {}, outputs

    def _loss_keys(self):
        """"""
        return self.apply_to if self.is_loss else []

    def _out_keys(self):
        """"""
        return [] if self.is_loss else self.apply_to

    def extra_repr(self):
        return c_f.extra_repr(self, ["apply_to"])


class TrueHook(BaseConditionHook):
    """Returns ```True```"""

    def call(self, losses, inputs):
        """"""
        return True


class FalseHook(BaseConditionHook):
    """Returns ```False```"""

    def call(self, losses, inputs):
        """"""
        return False


class NotHook(BaseConditionHook):
    """Returns the boolean negation of the wrapped hook."""

    def __init__(self, hook: BaseConditionHook, **kwargs):
        """
        Arguments:
            hook: The condition hook that will be negated.
        """
        super().__init__(**kwargs)
        self.hook = hook

    def call(self, losses, inputs):
        """"""
        return not self.hook(losses, inputs)


class AssertHook(BaseWrapperHook):
    """
    Asserts that the output keys of a hook match a specified regex string
    """

    def __init__(self, hook: BaseHook, allowed: str, **kwargs):
        """
        Arguments:
            hook: The wrapped hook
            allowed: The output dictionary of ```hook```
                must have keys that match the ```allowed``` regex.
        """
        super().__init__(**kwargs)
        self.hook = hook
        if not isinstance(allowed, str):
            raise TypeError("allowed must be a str")
        self.allowed = allowed

    def call(self, losses, inputs):
        """"""
        losses, outputs = self.hook(losses, inputs)
        self.assert_fn(outputs)
        return losses, outputs

    def assert_fn(self, outputs):
        filtered = c_f.filter(outputs, self.allowed)
        if len(filtered) != len(outputs):
            error_str = f"{c_f.cls_name(self.hook)} is producing outputs that don't match the allowed regex in {c_f.cls_name(self)}\n"
            error_str += f"output keys = {outputs.keys()}\n"
            error_str += f"regex filter = {self.allowed}"
            raise ValueError(error_str)

    def extra_repr(self):
        return c_f.extra_repr(self, ["allowed"])


class MultiplierHook(BaseWrapperHook):
    """
    Multiplies every loss by a scalar
    """

    def __init__(self, hook: BaseHook, m: float, **kwargs):
        """
        Arguments:
            hook: The losses of this hook will be multiplied by ```m```
            m: The scalar
        """
        super().__init__(**kwargs)
        self.hook = hook
        self.m = m

    def call(self, losses, inputs):
        """"""
        losses, outputs = self.hook(losses, inputs)
        losses = {k: v * self.m for k, v in losses.items()}
        return losses, outputs

    def extra_repr(self):
        return c_f.extra_repr(self, ["m"])


class RepeatHook(BaseHook):
    """
    Executes the wrapped hook ```n``` times.
    """

    def __init__(self, hook: BaseHook, n: int, keep_only_last: bool = False, **kwargs):
        """
        Arguments:
            hook: The hook that will be executed ```n``` times
            n: The number of times the hook will be executed.
            keep_only_last: If ```False```, the (losses, outputs) from each execution
                will be accumulated, and the keys will have the iteration number appended.
                If ```True```, then only the (losses, outputs) of the final execution will
                be kept.
        """
        super().__init__(**kwargs)
        self.hook = hook
        self.n = n
        self.keep_only_last = keep_only_last

    def call(self, losses, inputs):
        """"""
        losses, outputs = {}, {}
        for i in range(self.n):
            x = self.hook(losses, inputs)
            if self.keep_only_last and i == self.n - 1:
                losses, outputs = x
            else:
                losses.update({f"{k}{i}": v for k, v in x[0].items()})
                outputs.update({f"{k}{i}": v for k, v in x[1].items()})
        return losses, outputs

    def _loss_keys(self):
        """"""
        if self.keep_only_last:
            return self.hook.loss_keys
        else:
            return [f"{k}{i}" for k in self.hook.loss_keys for i in range(self.n)]

    def _out_keys(self):
        """"""
        if self.keep_only_last:
            return self.hook.out_keys
        else:
            return [f"{k}{i}" for k in self.hook.out_keys for i in range(self.n)]

    def extra_repr(self):
        return c_f.extra_repr(self, ["n", "keep_only_last"])


class ApplyToListHook(BaseWrapperHook):
    def __init__(self, hook, list_size, regex, **kwargs):
        super().__init__(**kwargs)
        self.hook = hook
        self.list_size = list_size
        self.regex = regex

    def call(self, losses, inputs):
        """"""
        search_strs = c_f.filter(inputs, self.regex)
        extracted = c_f.extract(inputs, search_strs)
        if len(set(len(x) for x in extracted)) != 1:
            raise TypeError(f"All extracted input lists must be equal in length")
        losses = {}
        for i in range(len(extracted[0])):
            curr_extracted = [x[i] for x in extracted]
            curr_extracted = {k: v for k, v in zip(search_strs, curr_extracted)}
            x = self.hook(losses, {**inputs, **curr_extracted})
            losses.update({f"{k}{i}": v for k, v in x[0].items()})
        return losses, {}

    def _loss_keys(self):
        """"""
        return [f"{k}{i}" for k in self.hook.loss_keys for i in range(self.list_size)]

    def _out_keys(self):
        """"""
        return []

    def extra_repr(self):
        return c_f.extra_repr(self, ["list_size", "regex"])

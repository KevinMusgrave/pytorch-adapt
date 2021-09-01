from typing import List, Union

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
    def __init__(self, loss_names, out_names, **kwargs):
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
        losses, outputs = {}, {}
        all_inputs = inputs
        prev_outputs = {}
        for i, h in enumerate(self.hooks):
            self.check_overwrite(i, all_inputs, prev_outputs)
            all_inputs = {**all_inputs, **prev_outputs}
            if self.conditions[i](losses, all_inputs):
                x = h(losses, all_inputs)
            else:
                x = self.alts[i](losses, all_inputs)
            prev_outputs = x[1]
            self.check_loss_overlap(losses, x[0])
            losses.update(x[0])
            outputs.update(prev_outputs)
        return losses, outputs

    def check_overlap(self, x, y, names):
        overlap = x.keys() & y.keys()
        if len(overlap) > 0:
            raise KeyError(
                f"overwrite is false, but {names[0]} and {names[1]} have overlapping keys: {overlap}"
            )

    def check_overwrite(self, i, kwargs, prev_outputs):
        if not self.overwrite or (
            isinstance(self.overwrite, list) and i not in self.overwrite
        ):
            self.check_overlap(kwargs, prev_outputs, ["kwargs", "prev_outputs"])

    def check_loss_overlap(self, losses, new_losses):
        self.check_overlap(losses, new_losses, ["losses", "new_losses"])

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
        losses, outputs = {}, {}
        for h in self.hooks:
            x = h(losses, inputs)
            losses.update(x[0])
            outputs.update(x[1])

        return losses, outputs

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


# Returns only outputs that are not present in kwargs
# You should use this if you want to change the value of
# a key passed to self.hook, but not propagate that change
# to the outside
class OnlyNewOutputsHook(BaseWrapperHook):
    def __init__(self, hook, **kwargs):
        super().__init__(**kwargs)
        self.hook = hook

    def call(self, losses, inputs):
        """"""
        losses, outputs = self.hook(losses, inputs)
        outputs = {k: outputs[k] for k in (outputs.keys() - inputs.keys())}
        return losses, outputs


class ApplyFnHook(BaseHook):
    def __init__(self, fn, apply_to, is_loss=False, **kwargs):
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
    def __init__(self, hook, allowed, **kwargs):
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
    def __init__(self, hook, m, **kwargs):
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
    def __init__(self, hook, n, keep_only_last=False, **kwargs):
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

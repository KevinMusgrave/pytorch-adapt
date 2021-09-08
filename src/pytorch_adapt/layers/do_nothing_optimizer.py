import torch


class DoNothingOptimizer(torch.optim.Optimizer):
    """
    An optimizer that doesn't do anything,
    i.e. ```step``` and ```zero_grad``` are empty functions.
    """

    def __init__(self, *arg, **kwargs):
        self.param_groups = [{"lr": 0}]

    def step(self):
        """"""
        pass

    def zero_grad(self):
        """"""
        pass

    def state_dict(self):
        """"""
        return {}

    def load_state_dict(self, state_dict):
        """"""
        pass

    def __repr__(self):
        return "DoNothingOptimizer()"

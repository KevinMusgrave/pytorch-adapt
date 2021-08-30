from collections import defaultdict

import torch
from ignite.engine import Engine
from ignite.metrics import Metric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class DictionaryAccumulator(Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    @reinit__is_reduced
    def reset(self):
        super().reset()
        self.accumulated = defaultdict(list)

    @reinit__is_reduced
    def update(self, output):
        for k, v in output.items():
            self.accumulated[k].append(v)

    @sync_all_reduce("accumulated")
    def compute(self):
        for k, v in self.accumulated.items():
            self.accumulated[k] = torch.cat(v, dim=0)
        return self.accumulated

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        self.update(engine.state.output)

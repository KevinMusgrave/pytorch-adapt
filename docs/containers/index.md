# Containers

Containers simplify object creation. 

## Examples
[Create with](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb)
```python
import torch

from pytorch_adapt.containers import LRSchedulers, Models, Optimizers

G = torch.nn.Linear(1000, 100)
C = torch.nn.Linear(100, 10)
D = torch.nn.Linear(100, 1)

models = Models({"G": G, "C": C, "D": D})
optimizers = Optimizers((torch.optim.Adam, {"lr": 0.456}))
schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.99}))

optimizers.create_with(models)
schedulers.create_with(optimizers)

# optimizers contains an optimizer for G, C, and D
# schedulers contains an LR scheduler for each optimizer

print(models)
print(optimizers)
print(schedulers)
```


[Merge](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb)
```python
more_models = Models({"X": torch.nn.Linear(20, 1)})
models.merge(more_models)

optimizers = Optimizers((torch.optim.Adam, {"lr": 0.456}))
special_opt = Optimizers((torch.optim.SGD, {"lr": 1}), keys=["G", "X"])
optimizers.merge(special_opt)
optimizers.create_with(models)

# models contains G, C, D, and X
# optimizers:
# - the Adam optimizer with lr 0.456 for models C and D
# - the SGD optimizer with lr 1 for models G and X

print(models)
print(optimizers)
```
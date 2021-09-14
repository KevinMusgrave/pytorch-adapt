# Hooks

Hooks are the building blocks of the algorithms in this library.

Every hook is a callable that takes in 2 arguments that represent the current context:

1. A dictionary of previously computed losses.
2. A dictionary of everything else that has been previously computed or passed in.

## Examples
[Computing Features](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb#Hooks-Basics)
```python
from pytorch_adapt.hooks import FeaturesHook

G = torch.nn.Linear(1000, 100)
models = {"G": G}
data = {
    "src_imgs": torch.randn(32, 1000),
    "target_imgs": torch.randn(32, 1000),
}

hook = FeaturesHook()

losses, outputs = hook({}, {**models, **data})
# outputs contains src_imgs_features and target_imgs_features
print(outputs.keys())

losses, outputs = hook({}, {**models, **data, **outputs})
# outputs is empty
print(outputs.keys())

hook = FeaturesHook(detach=True)
losses, outputs = hook({}, {**models, **data, **outputs})
# outputs contains
# src_imgs_features_detached and target_imgs_features_detached
print(outputs.keys())
```

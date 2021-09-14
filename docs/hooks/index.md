# Hooks

Hooks are the building blocks of the algorithms in this library.

## [Customizing Algorithms](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/CustomizingAlgorithms.ipynb)

This section shows examples of combining hooks to create complex algorithms.

### DANN
```python
from pytorch_adapt.hooks import DANNHook

hook = DANNHook(opts)
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
```

### DANN + MCC + ATDOC
```python
from pytorch_adapt.hooks import ATDOCHook, MCCHook

mcc = MCCHook()
atdoc = ATDOCHook(dataset_size=10000, feature_dim=100, num_classes=10)

hook = DANNHook(opts, post_g=[mcc, atdoc])
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
```

### CDAN
```python
from pytorch_adapt.hooks import CDANHook
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.utils import common_functions as c_f

d_opts = c_f.extract(optimizers, ["D"])
g_opts = c_f.extract(optimizers, ["G", "C"])
misc = {"feature_combiner": RandomizedDotProduct([100, 10], 100)}

hook = CDANHook(d_opts=d_opts, g_opts=g_opts)
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **misc, **data})
```

### CDAN + VAT
```python
from pytorch_adapt.hooks import VATHook

misc["combined_model"] = torch.nn.Sequential(G, C)
hook = CDANHook(d_opts=d_opts, g_opts=g_opts, post_g=[VATHook()])
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **misc, **data})
```

### MCD
```python
from pytorch_adapt.hooks import MCDHook
from pytorch_adapt.layers import MultipleModels

C = torch.nn.Linear(100, 10)
C1, C2 = C, c_f.reinit(C)
C = MultipleModels(C1, C2)
models["C"] = C

g_opts = c_f.extract(optimizers, ["G"])
c_opts = c_f.extract(optimizers, ["C"])

hook = MCDHook(g_opts=g_opts, c_opts=c_opts)
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
```

### MCD + AFN + MMD
```python
from pytorch_adapt.hooks import AFNHook, AlignerHook

hook = MCDHook(g_opts=g_opts, c_opts=c_opts, post_x=[AFNHook()], post_z=[AlignerHook()])
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
```



## [Building Blocks](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb)

Every hook is a callable that takes in 2 arguments that represent the current context:

1. A dictionary of previously computed losses.
2. A dictionary of everything else that has been previously computed or passed in.

### Computing Features
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

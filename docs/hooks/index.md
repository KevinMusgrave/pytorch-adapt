# Hooks

Hooks are the building blocks of the algorithms in this library.

## [Customizing Algorithms](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/CustomizingAlgorithms.ipynb)

This section shows examples of combining hooks to create complex algorithms.

### Create models, optimizers, data etc.
```python
import torch

from pytorch_adapt.datasets import get_mnist_mnistm
from pytorch_adapt.hooks import validate_hook
from pytorch_adapt.models import Discriminator, mnistC, mnistG

# mnist is the source domain
# mnistm is the target domain
datasets = get_mnist_mnistm(["mnist"], ["mnistm"], ".", download=True)
dataloader = torch.utils.data.DataLoader(
    datasets["train"], batch_size=32, num_workers=2
)
data = iter(dataloader).next()

G = mnistG(pretrained=True)
C = mnistC(pretrained=True)
D = Discriminator(in_size=1200, h=256)
models = {"G": G, "C": C, "D": D}

G_opt = torch.optim.Adam(G.parameters(), lr=0.0001)
C_opt = torch.optim.Adam(C.parameters(), lr=0.0001)
D_opt = torch.optim.Adam(D.parameters(), lr=0.0001)
opts = [G_opt, C_opt, D_opt]
```

### Source Classifier
```python
from pytorch_adapt.hooks import ClassifierHook

hook = ClassifierHook(opts)
model_counts = validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
print(f"Expected model counts = {dict(model_counts)}")
# Should print {"G": 1, "C": 1}
```

### Source Classifier + BSP + BNM
```python
from pytorch_adapt.hooks import BNMHook, BSPHook
from pytorch_adapt.weighters import MeanWeighter

weighter = MeanWeighter(weights={"bsp_loss": 1e-5})
hook = ClassifierHook(opts, post=[BSPHook(), BNMHook()], weighter=weighter)
model_counts = validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
print(f"Expected model counts = {dict(model_counts)}")
# Should print {"G": 2, "C": 2}
```

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
atdoc = ATDOCHook(dataset_size=len(datasets["train"]), feature_dim=1200, num_classes=10)

hook = DANNHook(opts, post_g=[mcc, atdoc])
validate_hook(hook, list(data.keys()))
losses, outputs = hook({}, {**models, **data})
```

### CDAN
```python
from pytorch_adapt.hooks import CDANHook
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.utils import common_functions as c_f

d_opts = opts[2:]
g_opts = opts[:2]
misc = {"feature_combiner": RandomizedDotProduct([1200, 10], 1200)}

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

C2 = c_f.reinit(C)
C = MultipleModels(C, C2)
models["C"] = C

g_opts = opts[0:1]
c_opts = opts[1:2]

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

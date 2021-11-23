# Adapters

Adapters contain an algorithm's training step and inference step. The training step is defined in the wrapped [hook](../hooks/index.md). 

## [Examples](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb)
### Initialization
```python
import torch

from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models

G = torch.nn.Linear(1000, 100)
C = torch.nn.Linear(100, 10)
D = torch.nn.Sequential(torch.nn.Linear(100, 1), torch.nn.Flatten(start_dim=0))
models = Models({"G": G, "C": C, "D": D})

adapter = DANN(models=models)
```

### Training step
```python
from pytorch_adapt.utils import common_functions as c_f

device = torch.device("cuda")
adapter.models.to(device)

data = {
    "src_imgs": torch.randn(32, 1000),
    "target_imgs": torch.randn(32, 1000),
    "src_labels": torch.randint(0, 10, size=(32,)),
    "src_domain": torch.zeros(32),
    "target_domain": torch.zeros(32),
}

data = c_f.batch_to_device(data, device)
loss = adapter.training_step(data)
```

### Inference
```python
data = torch.randn(32, 1000).to(device)
features, logits = adapter.inference(data)
```
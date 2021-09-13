# Adapters

Adapters contain an algorithm's training step and inference step. The training step is defined in the wrapped [hook](../hooks/index.md). 

## Examples
[Initialization](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb#adapters.index.md-initialization)
```python
import torch

from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models

G = torch.nn.Linear(1000, 100)
C = torch.nn.Linear(100, 10)
D = torch.nn.Linear(100, 1)
models = Models({"G": G, "C": C, "D": D})

adapter = DANN(models=models)
```

[Training step](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb#adapters.index.md-training-step)
```python
device = torch.device("cuda")
adapter.models.to(device)

data = {
    "src_imgs": torch.randn(32, 1000),
    "target_imgs": torch.randn(32, 1000),
    "src_labels": torch.randint(0, 10, size=(32,)),
    "src_domain": torch.zeros(32),
    "target_domain": torch.zeros(32),
}

loss = adapter.training_step(data, device)
```

[Inference](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb#adapters.index.md-inference)
```python
data = torch.randn(32, 1000).to(device)
features, logits = adapter.inference(data)
```
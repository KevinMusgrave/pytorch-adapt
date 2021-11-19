# Frameworks

This library works with plain PyTorch, but what about PyTorch frameworks like [Ignite](https://github.com/pytorch/ignite), [Lightning](https://github.com/PyTorchLightning/pytorch-lightning), and [Catalyst](https://github.com/catalyst-team/catalyst)?

Of course, anything that works in PyTorch will also work in PyTorch frameworks. However, there is some setup involved, like registering event handlers in Ignite, or writing class definitions in Lightning and Catalyst.

The purpose of the ```frameworks``` module is to eliminate that setup.

## [Examples](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/DANNIgnite.ipynb)
### Ignite
```python
import torch

from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm
from pytorch_adapt.frameworks import Ignite
from pytorch_adapt.models import Classifier, Discriminator, MNISTFeatures
from pytorch_adapt.validators import IMValidator

G = MNISTFeatures()
C = Classifier(num_classes=10, in_size=1200, h=256)
D = Discriminator(in_size=1200, h=256)
models = Models({"G": G, "C": C, "D": D})
optimizers = Optimizers((torch.optim.Adam, {"lr": 0.123}))

adapter = DANN(models=models, optimizers=optimizers)
wrapped_adapter = Ignite(adapter)
validator = IMValidator()

datasets = get_mnist_mnistm(["mnist"], ["mnistm"], folder=".", download=True)
dc = DataloaderCreator(batch_size=32, num_workers=2)

wrapped_adapter.run(datasets, dataloader_creator=dc, validator=validator, max_epochs=2)
```
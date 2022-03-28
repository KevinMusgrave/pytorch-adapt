# PyTorch Adapt

## Getting started
See the **[examples folder](https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/README.md)** for notebooks you can download or run on Google Colab.

## How to...

### Use in vanilla PyTorch
```python
from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.utils.common_functions import batch_to_device

# Assuming that models, optimizers, and dataloader are already created.
hook = DANNHook(optimizers)
for data in tqdm(dataloader):
    data = batch_to_device(data, device)
    # Optimization is done inside the hook.
    # The returned loss is for logging.
    loss, _ = hook({}, {**models, **data})
```

### Build complex algorithms
Let's customize ```DANNHook``` with:

- minimum class confusion
- virtual adversarial training

```python
from pytorch_adapt.hooks import MCCHook, VATHook

# G and C are the Generator and Classifier models
G, C = models["G"], models["C"]
misc = {"combined_model": torch.nn.Sequential(G, C)}
hook = DANNHook(optimizers, post_g=[MCCHook(), VATHook()])
for data in tqdm(dataloader):
    data = batch_to_device(data, device)
    loss, _ = hook({}, {**models, **data, **misc})
```

### Wrap with your favorite PyTorch framework
First, set up the adapter and dataloaders:

```python
from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models
from pytorch_adapt.datasets import DataloaderCreator

models_cont = Models(models)
adapter = DANN(models=models_cont)
dc = DataloaderCreator(num_workers=2)
dataloaders = dc(**datasets)
```

Then use a framework wrapper:

#### PyTorch Lightning
```python
import pytorch_lightning as pl
from pytorch_adapt.frameworks.lightning import Lightning

L_adapter = Lightning(adapter)
trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(L_adapter, dataloaders["train"])
```

#### PyTorch Ignite
```python
trainer = Ignite(adapter)
trainer.run(datasets, dataloader_creator=dc)
```

### Check your model's performance
You can do this in vanilla PyTorch:
```python
from pytorch_adapt.validators import SNDValidator

# Assuming predictions have been collected
target_train = {"preds": preds}
validator = SNDValidator()
score = validator(target_train=target_train)
```

You can also do this during training with a framework wrapper:

#### PyTorch Lightning
```python
from pytorch_adapt.frameworks.utils import filter_datasets

validator = SNDValidator()
dataloaders = dc(**filter_datasets(datasets, validator))
train_loader = dataloaders.pop("train")

L_adapter = Lightning(adapter, validator=validator)
trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(L_adapter, train_loader, list(dataloaders.values()))
```

#### Pytorch Ignite
```python
from pytorch_adapt.validators import ScoreHistory

validator = ScoreHistory(SNDValidator())
trainer = Ignite(adapter, validator=validator)
trainer.run(datasets, dataloader_creator=dc)
```

### Run the above examples
See [this notebook](https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/other/ReadmeExamples.ipynb) and [the examples page](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/) for other notebooks.

## Installation

### Pip
```
pip install pytorch-adapt
```

**To get the latest dev version**:
```
pip install pytorch-adapt --pre
```

**To use ```pytorch_adapt.frameworks.lightning```**:
```
pip install pytorch-adapt[lightning]
```

**To use ```pytorch_adapt.frameworks.ignite```**:
```
pip install pytorch-adapt[ignite]
```


### Conda
Coming soon...

### Dependencies
Required dependencies:

- numpy
- torch >= 1.6
- torchvision
- torchmetrics
- pytorch-metric-learning >= 1.0.0.dev5
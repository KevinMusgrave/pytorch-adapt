<h1 align="center">
<a href="https://github.com/KevinMusgrave/pytorch-adapt">
<img alt="Logo" src="https://github.com/KevinMusgrave/pytorch-adapt/blob/main/docs/imgs/Logo.png">
</a>
</h2>
<p align="center">
 <a href="https://badge.fury.io/py/pytorch-adapt">
     <img alt="PyPi version" src="https://badge.fury.io/py/pytorch-adapt.svg">
 </a> 
</p>

## News
**November 19**: Git repo is now public

## Documentation
- [**Documentation**](https://kevinmusgrave.github.io/pytorch-adapt/)
- [**Installation instructions**](https://github.com/KevinMusgrave/pytorch-adapt#installation)
- [**List of algorithms/papers implemented**](https://kevinmusgrave.github.io/pytorch-adapt/algorithms/uda)

## Google Colab Examples
See the **[examples folder](https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/README.md)** for notebooks you can download or run on Google Colab.
  
## Overview
This library consists of 11 modules:

| Module | Description |
| --- | --- |
| [**Adapters**](https://kevinmusgrave.github.io/pytorch-adapt/adapters) | Wrappers for training and inference steps
| [**Containers**](https://kevinmusgrave.github.io/pytorch-adapt/containers) | Dictionaries for simplifying object creation
| [**Datasets**](https://kevinmusgrave.github.io/pytorch-adapt/datasets) | Commonly used datasets and tools for domain adaptation
| [**Frameworks**](https://kevinmusgrave.github.io/pytorch-adapt/frameworks) | Wrappers for training/testing pipelines
| [**Hooks**](https://kevinmusgrave.github.io/pytorch-adapt/hooks) | Modular building blocks for domain adaptation algorithms
| [**Layers**](https://kevinmusgrave.github.io/pytorch-adapt/layers) | Loss functions and helper layers
| [**Meta Validators**](https://kevinmusgrave.github.io/pytorch-adapt/meta_validators) | Post-processing of metrics, for hyperparameter optimization
| [**Models**](https://kevinmusgrave.github.io/pytorch-adapt/models) | Architectures used for benchmarking and in examples
| [**Utils**](https://kevinmusgrave.github.io/pytorch-adapt/utils) | Various tools
| [**Validators**](https://kevinmusgrave.github.io/pytorch-adapt/validators) | Metrics for determining and estimating accuracy
| [**Weighters**](https://kevinmusgrave.github.io/pytorch-adapt/weighters) | Functions for weighting losses

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

- virtual adversarial training
- entropy conditioning

```python
from pytorch_adapt.hooks import EntropyReducer, MeanReducer, VATHook

# G and C are the Generator and Classifier models
misc = {"combined_model": torch.nn.Sequential(G, C)}
reducer = EntropyReducer(
    apply_to=["src_domain_loss", "target_domain_loss"], default_reducer=MeanReducer()
)
hook = DANNHook(optimizers, reducer=reducer, post_g=[VATHook()])
for data in tqdm(dataloader):
    data = batch_to_device(data, device)
    loss, _ = hook({}, {**models, **data, **misc})
```

### Wrap with your favorite PyTorch framework
For additional functionality, adapters can be wrapped with a framework (currently just PyTorch Ignite). 
```python
from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.frameworks.ignite import Ignite

# Assume G, C and D are existing models
models_cont = Models(models)
# Override the default optimizer for G and C
optimizers_cont = Optimizers((torch.optim.Adam, {"lr": 0.123}), keys=["G", "C"])
adapter = DANN(models=models_cont, optimizers=optimizers_cont)

dc = DataloaderCreator(num_workers=2)
trainer = Ignite(adapter)
trainer.run(datasets, dataloader_creator=dc)
```
Wrappers for other frameworks (e.g. PyTorch Lightning and Catalyst) are planned to be added.

### Check your model's performance
You can do this in vanilla PyTorch:
```python
from pytorch_adapt.validators import SNDValidator

# Assuming predictions have been collected
target_train = {"preds": preds}
validator = SNDValidator()
score = validator.score(epoch=1, target_train=target_train)
```

You can also do this using a framework wrapper:
```python
validator = SNDValidator()
trainer = Ignite(adapter, validator=validator)
trainer.run(datasets, dataloader_creator=dc)
```

### Run the above examples
See [this notebook](https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/notebooks/ReadmeExamples.ipynb) and [the examples page](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/) for other notebooks.

## Installation

### Pip
```
pip install pytorch-adapt
```

**To get the latest dev version**:
```
pip install pytorch-adapt --pre
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

## Acknowledgements

### Contributors
Pull requests are welcome!

### Advisors
Thank you to [Ser-Nam Lim](https://research.fb.com/people/lim-ser-nam/), and my research advisor, [Professor Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/).

### Logo
Thanks to [Jeff Musgrave](https://www.designgenius.ca/) for designing the logo.

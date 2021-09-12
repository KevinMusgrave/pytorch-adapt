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

## Documentation
- [**View the documentation here**](https://kevinmusgrave.github.io/pytorch-adapt/)
- [**View the installation instructions here**](https://github.com/KevinMusgrave/pytorch-adapt#installation)

## Google Colab Examples
See the [examples folder](https://github.com/KevinMusgrave/pytorch-adapt/blob/master/examples/README.md) for notebooks you can download or run on Google Colab.
  
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
 
## How to use

### Use in vanilla PyTorch
```python
from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.utils.common_functions import batch_to_device

# Assuming that models, optimizers, and dataloader are already created.
hook = DANNHook(optimizers)
for data in dataloader:
    data = batch_to_device(data, device)
    # Optimization is done inside the hook.
    # The returned loss is for logging.
    loss, _ = hook({}, {**models, **data})
```

### Build complex algorithms
Let's customize ```DANNHook``` with the following:
- virtual adversarial training
- entropy conditioning

```python
from pytorch_adapt.hooks import EntropyReducer, MeanReducer, VATHook

# G and C are the Generator and Classifier models
models["combined_model"] = torch.nn.Sequential(G, C)
reducer = EntropyReducer(
    apply_to=["src_domain_loss", "target_domain_loss"], default_reducer=MeanReducer()
)
hook = DANNHook(opts, reducer=reducer, post_g=[VATHook()])

# then loop through the dataloader as shown above
```

## Installation

### Pip
```
pip install pytorch-adapt
```

**To get the latest dev version**:
```
pip install pytorch-adapt --pre
```

### Conda
Coming soon...

### Dependencies
Coming soon...

## Acknowledgements

### Contributors
Pull requests are welcome!

### Logo
Thanks to [Jeff Musgrave](https://jeffmusgrave.com) for designing the logo.

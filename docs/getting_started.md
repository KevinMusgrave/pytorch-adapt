# Getting started

## Examples

Currently the best place to start is the **[example jupyter notebooks](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples)**. You can download them or run them on Google Colab.

After that, you might be interested in looking at the documentation. This is a large library, so some of the documentation isn't filled in yet. I'm currently focused on adding notebook examples, because I think they provide an easier introduction to this library.

Please **[open an issue on GitHub](https://github.com/KevinMusgrave/pytorch-adapt/issues)** if you have any questions.


## Overview
This library consists of 12 modules:

| Module | Description |
| --- | --- |
| [**Adapters**](docs/adapters/index.md) | Wrappers for training and inference steps
| [**Containers**](docs/containers/index.md) | Dictionaries for simplifying object creation
| [**Datasets**](docs/datasets/index.md) | Commonly used datasets and tools for domain adaptation
| [**Frameworks**](docs/frameworks/index.md) | Wrappers for training/testing pipelines
| [**Hooks**](docs/hooks/index.md) | Modular building blocks for domain adaptation algorithms
| [**Inference**](docs/inference/index.md) | Algorithm-specific functions used during inference
| [**Layers**](docs/layers/index.md) | Loss functions and helper layers
| [**Meta Validators**](docs/meta_validators/index.md) | Post-processing of metrics, for hyperparameter optimization
| [**Models**](docs/models/index.md) | Architectures used for benchmarking and in examples
| [**Utils**](docs/utils/index.md) | Various tools
| [**Validators**](docs/validators/index.md) | Metrics for determining and estimating accuracy
| [**Weighters**](docs/weighters/index.md) | Functions for weighting losses


---

## Adapters

Adapters contain an algorithm's training step and inference step. The training step is defined in the wrapped [hook](#hooks). 

View **[this notebook for examples](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/in_depth/Adapters.ipynb)**.

---

## Containers

Containers are Python dictionaries with extra functions that simplify object creation. 

View **[this notebook for examples](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/in_depth/Containers.ipynb)**.

---

## Datasets

The datasets module consists of wrapper classes that output data in a format compatible with [hooks](#hooks).

It also contains some common domain-adaptation datasets like [MNISTM][pytorch_adapt.datasets.MNISTM], [Office31][pytorch_adapt.datasets.Office31], and [OfficeHome][pytorch_adapt.datasets.OfficeHome].

View **[this notebook for examples](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/in_depth/Datasets.ipynb)**.

---

## Frameworks

This library works with plain PyTorch, but what about PyTorch frameworks like [Ignite](https://github.com/pytorch/ignite), [Lightning](https://github.com/PyTorchLightning/pytorch-lightning), and [Catalyst](https://github.com/catalyst-team/catalyst)?

Of course, anything that works in PyTorch will also work in PyTorch frameworks. However, there is some setup involved, like registering event handlers in Ignite, or writing class definitions in Lightning and Catalyst.

The purpose of the ```frameworks``` module is to eliminate that setup.

Check out these notebooks for examples:

- **[DANN with PyTorch Lightning](https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/getting_started/DANNLightning.ipynb)**
- **[DANN with PyTorch Ignite](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/getting_started/DANNIgnite.ipynb)**
- **[DANN with PyTorch Ignite + Visualizations](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/getting_started/DANNIgniteWithViz.ipynb)**

---

## Hooks

Hooks are the building blocks of the algorithms in this library.

Check out these notebooks for examples:

- **[Customizing Algorithms](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/getting_started/CustomizingAlgorithms.ipynb)**
- **[Hooks explained in depth](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/in_depth/Hooks.ipynb)**

---

## Layers

The layers module contains loss functions and wrapper models. These are used in combination with hooks to create domain adaptation algorithms.

---

## Validators

Validators compute or estimate the accuracy of a model.

---

## Weighters

Weighters multiply losses by scalar values, and then reduce the losses to a single value on which you call ```.backward()```.

View **[this notebook for examples](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/in_depth/Weighters.ipynb)**.
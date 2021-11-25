# Frameworks

This library works with plain PyTorch, but what about PyTorch frameworks like [Ignite](https://github.com/pytorch/ignite), [Lightning](https://github.com/PyTorchLightning/pytorch-lightning), and [Catalyst](https://github.com/catalyst-team/catalyst)?

Of course, anything that works in PyTorch will also work in PyTorch frameworks. However, there is some setup involved, like registering event handlers in Ignite, or writing class definitions in Lightning and Catalyst.

The purpose of the ```frameworks``` module is to eliminate that setup.

Check out these notebooks for examples:

- **[DANN with PyTorch Ignite](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/getting_started/DANNIgnite.ipynb)**
- **[DANN with PyTorch Ignite + Visualizations](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/getting_started/DANNIgniteWithViz.ipynb)**
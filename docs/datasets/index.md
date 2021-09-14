# Datasets

The datasets module consists of wrapper classes that output data in a format compatible with [hooks](../hooks/index.md).

It also contains some common domain-adaptation datasets like [MNISTM](mnistm.md), [Office31](office31.md), and [DomainNet](domainnet.md).

## Examples
[Source and target datasets](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb#datasets.index.md-source-and-target-datasets)
```python
from torchvision.datasets import MNIST

from pytorch_adapt.datasets import (
    MNISTM,
    CombinedSourceAndTargetDataset,
    SourceDataset,
    TargetDataset,
)

x = MNIST(root=".", train=True, transform=None)
y = MNISTM(root=".", train=True, transform=None)
# x and y return (data, label) tuples
print(x[0])
print(y[0])

x = SourceDataset(x)
y = TargetDataset(y)
# x and y return dictionaries
print(x[0])
print(y[0])

xy = CombinedSourceAndTargetDataset(x, y)
# xy returns a dictionary
print(xy[0])
```


[Getters and DataloaderCreator](https://github.com/KevinMusgrave/pytorch-adapt/tree/main/examples/notebooks/docs_examples.ipynb#datasets.index.md-getters-and-DataloaderCreator)
```python
from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm

datasets = get_mnist_mnistm(["mnist"], ["mnistm"], folder=".")
dc = DataloaderCreator(batch_size=128)
dataloaders = dc(**datasets)

# datasets and dataloaders are dictionaries
print(datasets)
print(dataloaders)
```
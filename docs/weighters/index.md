# Weighters

Weighters multiply losses by scalar values, and then reduce the losses to a single value on which you call ```.backward()```.

For example:
```python
import torch
from pytorch_adapt.weighters import MeanWeighter

weighter = MeanWeighter(weights={"y": 2.3})

logits = torch.randn(32,512)
labels = torch.randint(0, 10, size=(32,))

x = torch.nn.functional.cross_entropy(logits, labels)
y = torch.norm(logits)

# y will by multiplied by 2.3
# x wasn't given a weight, 
# so it gets multiplied by the default value of 1.
loss, components = weighter({"x": x, "y": y})
loss.backward()
```
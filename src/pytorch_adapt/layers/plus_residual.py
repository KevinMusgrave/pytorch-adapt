import torch


class PlusResidual(torch.nn.Module):
    """
    Wraps a layer such that the forward pass returns
    ```x + self.layer(x)```
    """

    def __init__(self, layer: torch.nn.Module):
        """
        Arguments:
            layer: The layer to be wrapped.
        """
        super().__init__()
        self.layer = layer

    def forward(self, x):
        """"""
        return x + self.layer(x)

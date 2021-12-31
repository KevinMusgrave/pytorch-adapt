from .base_container import BaseContainer


class Models(BaseContainer):
    """
    A container with some functions specific to models
    that have optimizable parameters.
    """

    def train(self):
        """
        Sets all models to train mode.
        """
        for v in self.values():
            v.train()

    def eval(self):
        """
        Sets all models to eval mode.
        """
        for v in self.values():
            v.eval()

    def zero_grad(self):
        """
        Zeros the gradients in all models.
        """
        for v in self.values():
            v.zero_grad()

    def to(self, device):
        """
        Moves all models to ```device```.
        """
        for v in self.values():
            v.to(device)

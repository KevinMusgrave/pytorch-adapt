from .base_weighter import BaseWeighter


class SumWeighter(BaseWeighter):
    """
    Weights the losses and then returns the **sum** of the weighted losses.
    """

    def __init__(self, **kwargs):
        super().__init__(reduction=sum, **kwargs)

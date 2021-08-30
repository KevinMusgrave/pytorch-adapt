from .base_weighter import BaseWeighter


class SumWeighter(BaseWeighter):
    def __init__(self, **kwargs):
        super().__init__(reduction=sum, **kwargs)

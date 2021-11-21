from typing import Tuple


class ForwardOnlyValidator:
    """
    This is basically a pass-through function.
    It returns the best score and best epoch
    that is returned by the inner adapter.
    """

    def run(self, adapter, **kwargs) -> Tuple[float, int]:
        """
        Arguments:
            adapter: the framework-wrapped adapter.
            **kwargs: keyword arguments to be passed into adapter.run()
        Returns:
            the best score and best epoch
        """
        if not adapter.validator:
            raise KeyError(
                "An adapter validator is required when using ForwardOnlyValidator"
            )
        return adapter.run(**kwargs)

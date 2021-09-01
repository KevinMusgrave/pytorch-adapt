# Base

## BaseHook

All hooks extend ```BaseHook```.

``` python
class BaseHook(ABC):
    def __init__(
        self,
        loss_prefix="",
        loss_suffix="",
        out_prefix="",
        out_suffix="",
        key_map=None,
    )
```

### Abstract methods
``` python
@abstractmethod
def call(self, losses, inputs):
    pass

@abstractmethod
def _loss_keys(self):
    pass

@abstractmethod
def _out_keys(self):
    pass
```



## BaseConditionHook
The base class for hooks that return a boolean.


## BaseWrapperHook
A simple wrapper for calling ```self.hook```, which should be defined in the child's ```__init__``` function.
# Utils

## ChainHook

Calls multiple hooks sequentially. The Nth hook receives the context accumulated through hooks 0 to N-1.

```python
class ChainHook(BaseHook):
    def __init__(
        self,
        *hooks,
        conditions=None,
        alts=None,
        overwrite=False,
        **kwargs,
    )
```

## ParallelHook

Calls multiple hooks while keeping contexts separate. The Nth hook receives the same context as hooks 0 to N-1. All the output contexts are merged at the end.


```python
class ParallelHook(BaseHook):
    def __init__(self, *hooks, **kwargs)
```


## EmptyHook

Returns two empty dictionaries.

```python
class EmptyHook(BaseHook):
    def __init__(self, **kwargs)
```


## TrueHook

Returns ```True```

```python
class TrueHook(BaseConditionHook)
    def __init__(self, **kwargs)
```

## FalseHook

Returns ```False```

```python
class FalseHook(BaseConditionHook)
    def __init__(self, **kwargs)
```

## NotHook

Returns the boolean negation of the wrapped hook.

```python
class NotHook(BaseConditionHook):
    def __init__(self, hook, **kwargs)
```

# Hooks

Hooks are the main building block of this library.

Every hook is a callable that takes in 2 arguments that represent the current context:

1. A dictionary of previously computed losses.
2. A dictionary of everything else that has been previously computed or passed in.

The purpose of the context is to compute data only when necessary. 

For example, to compute a classification loss, a hook will need logits. If these logits are not available in the context, then they are computed, added to the context, and then used to compute the loss. If they are already in the context, then only the loss is computed.
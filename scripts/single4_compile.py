"""Preserve recurrent carry across CUDA graph iterations and EMA copies."""
import dataclasses
import torch
from torch import nn


def own_tensors(value):
    # Clone preserves gradients for the loss; detached carry remains detached.
    # Outputs retained by the driver must not alias reusable CUDA-graph storage.
    if torch.is_tensor(value): return value.clone()
    if dataclasses.is_dataclass(value):
        return type(value)(**{f.name: own_tensors(getattr(value, f.name)) for f in dataclasses.fields(value)})
    if isinstance(value, dict): return {k: own_tensors(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(own_tensors(v) for v in value)
    if isinstance(value, list): return [own_tensors(v) for v in value]
    return value


class ManagedCompile(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self._orig_mod = model
        self.mode = mode
        object.__setattr__(self, '_compiled_call', None)

    def __getstate__(self):
        # EMA deep-copies the model. Never copy a callable bound to old weights
        # or CUDA-graph allocations into the new evaluation model.
        state = super().__getstate__()
        state['_compiled_call'] = None
        return state

    def initial_carry(self, batch): return self._orig_mod.initial_carry(batch)

    @property
    def model(self): return self._orig_mod.model

    def forward(self, *args, **kwargs):
        if self._compiled_call is None:
            call = self._orig_mod if self.mode == 'eager' else torch.compile(self._orig_mod, mode=self.mode)
            object.__setattr__(self, '_compiled_call', call)
        if self.mode != 'eager': torch.compiler.cudagraph_mark_step_begin()
        return own_tensors(self._compiled_call(*args, **kwargs))

"""Depth-matched tied/untied attention unrolls with identical gradient horizons."""
import copy
import torch
from torch import nn
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1 as Native,
    TinyRecursiveReasoningModel_ACTV1Config as Config,
    TinyRecursiveReasoningModel_ACTV1_Inner as NativeInner,
    TinyRecursiveReasoningModel_ACTV1InnerCarry as Carry,
)


class Reasoning(nn.Module):
    def __init__(self, original, calls, untied):
        super().__init__()
        self.untied = untied
        # Identical initial functions in both arms; independent storage in the
        # untied arm. No extra RNG draws or changes to the attention block.
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(calls if untied else 1)
                                     for layer in original.layers])

    def forward(self, hidden_states, input_injection, call_index, **kwargs):
        hidden_states = hidden_states + input_injection
        start = 2*call_index if self.untied else 0
        for i in range(start, start+2):
            hidden_states = self.layers[i](hidden_states=hidden_states, **kwargs)
        return hidden_states


class Inner(NativeInner):
    def __init__(self, config, untied):
        super().__init__(config)
        self.L_level = Reasoning(self.L_level, config.H_cycles*(config.L_cycles+1), untied)

    def forward(self, carry, batch):
        seq = dict(cos_sin=self.rotary_emb() if hasattr(self, 'rotary_emb') else None)
        embedding = self._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])
        y, z = carry.z_H, carry.z_L
        call = 0
        # Both arms differentiate through all T*(N+1) calls. Retaining the
        # native no_grad prefix would leave early untied weights untrainable.
        for _ in range(self.config.H_cycles):
            for _ in range(self.config.L_cycles):
                z = self.L_level(z, y+embedding, call_index=call, **seq); call += 1
            y = self.L_level(y, z, call_index=call, **seq); call += 1
        logits = self.lm_head(y)[:, self.puzzle_emb_len:]
        q = self.q_head(y[:, 0]).float()
        return Carry(z_H=y.detach(), z_L=z.detach()), logits, (q[..., 0], q[..., 1])


class Model(Native):
    def __init__(self, config_dict):
        nn.Module.__init__(self)
        self.config = Config(**config_dict)
        if self.config.mlp_t or self.config.L_layers != 2:
            raise ValueError('Controls require two-layer native attention blocks.')
        self.inner = Inner(self.config, untied=config_dict['untied'])

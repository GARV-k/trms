"""Observe real native updates; never add a backward pass or alter gradients.

Selected updates execute the SAME loss module eagerly, allowing hooks on internal
states that otherwise disappear inside a compiled graph. Other updates compile
normally. This is an execution difference, not bitwise equivalence to compilation.
"""
import math
import torch
import torch.distributed as dist


def square_sum(x):
    return x.detach().float().square().sum(dtype=torch.float64)


def unpack(values):
    sq, count = values.tolist()
    return dict(l2=math.sqrt(sq), rms=math.sqrt(sq/max(count, 1)), elements=int(count))


class UpdateObserver:
    def __init__(self, model, optimizer, global_batch_size, rank=0, world=1):
        self.model = model
        self.inner = model.model.inner
        self.optimizer = optimizer
        self.batch = global_batch_size
        self.rank, self.world = rank, world
        cfg = self.inner.config
        if cfg.mlp_t or cfg.L_layers != 2 or not cfg.no_ACT_continue:
            raise ValueError('Diagnostics require the native two-layer attention/one-forward ACT recipe.')
        self.n, self.t = cfg.L_cycles, cfg.H_cycles
        self.calls = 0
        self.origins, self.states, self.gradients, self.last_delta = {}, {}, {}, {}
        self.layers = [list(layer.parameters()) for layer in self.inner.L_level.layers]
        self.handles = []
        self.layer_metrics = []

    def __enter__(self):
        self.handles.append(self.inner.L_level.register_forward_hook(self.forward_hook, with_kwargs=True))
        self.handles.append(self.optimizer.register_step_pre_hook(self.before_optimizer))
        self.handles.append(self.optimizer.register_step_post_hook(self.after_optimizer))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def forward_hook(self, module, args, kwargs, output):
        index = self.calls
        self.calls += 1
        name = 'y' if index % (self.n+1) == self.n else 'z'
        before = args[0] if args else kwargs['hidden_states']
        # Initial states are measured AFTER ACT reset, so resets are not mistaken
        # for reasoning updates. Native core has no in-place state mutations.
        if name not in self.origins:
            self.origins[name] = before.detach()
        if index in (self.t*(self.n+1)-2, self.t*(self.n+1)-1):
            if not output.requires_grad:
                raise RuntimeError('Final native cycle must have gradients enabled.')
            self.states[name] = output.detach()
            with torch.no_grad():
                self.last_delta[name] = self.stat(output.float()-before.float())
            def capture(gradient, key=name):
                # Undo native 1/global_batch loss scaling. This reports gradients
                # of each example's token-mean LM + 0.5*halt loss, comparable across
                # GPU counts. BF16 gradients are converted to FP32 for norms.
                self.gradients[key] = self.stat(gradient.float()*self.batch)
            output.register_hook(capture)

    @staticmethod
    def stat(tensor):
        return torch.stack((square_sum(tensor), tensor.new_tensor(tensor.numel(), dtype=torch.float64)))

    def before_optimizer(self, optimizer, args, kwargs):
        if self.rank != 0:
            return
        self.before = [[p.detach().clone() for p in layer] for layer in self.layers]
        for layer in self.layers:
            if any(p.grad is None for p in layer):
                raise RuntimeError('A native layer parameter has no gradient.')
            sq = sum(square_sum(p.grad) for p in layer).item()
            count = sum(p.numel() for p in layer)
            weight = math.sqrt(sum(square_sum(p) for p in layer).item())
            self.layer_metrics.append(dict(grad_l2=math.sqrt(sq), grad_rms=math.sqrt(sq/count),
                                           weight_l2_before=weight, parameters=count))

    def after_optimizer(self, optimizer, args, kwargs):
        if self.rank != 0:
            return
        for params, old, metrics in zip(self.layers, self.before, self.layer_metrics):
            delta = math.sqrt(sum(square_sum(p.detach().float()-v.float()) for p, v in zip(params, old)).item())
            metrics.update(update_l2=delta, update_rms=delta/math.sqrt(metrics['parameters']),
                           relative_update=delta/max(metrics['weight_l2_before'], 1e-30))
        del self.before

    def finish(self):
        if self.calls != self.t*(self.n+1) or set(self.gradients) != {'z', 'y'}:
            raise RuntimeError('Unexpected recurrence call count or missing latent gradients.')
        result = dict(latents={}, layers={f'layer_{i+1}': v for i, v in enumerate(self.layer_metrics)})
        for name in ('z', 'y'):
            stats = dict(gradient=self.gradients[name], state=self.stat(self.states[name]),
                         forward_change=self.stat(self.states[name].float()-self.origins[name].float()),
                         last_substep_change=self.last_delta[name])
            result['latents'][name] = {}
            for key, value in stats.items():
                if self.world > 1:
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                result['latents'][name][key] = unpack(value)
        return result


def observed_update(upstream, config, state, batch, effective_batch, rank, world):
    compiled = state.model
    eager = getattr(compiled, '_orig_mod', compiled)
    # The dense optimizer is last in the unchanged upstream optimizer list.
    observer = UpdateObserver(eager, state.optimizers[-1], effective_batch, rank, world)
    try:
        state.model = eager
        with observer:
            metrics = upstream.train_batch(config, state, batch, effective_batch, rank, world)
        diagnostics = observer.finish()
    finally:
        state.model = compiled
    return metrics, diagnostics

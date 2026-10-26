"""Full native attention recipe with resumable sampling/evaluation and diagnostics."""
import argparse
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from study.bootstrap import ROOT, VENDOR
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import pretrain as upstream
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.ema import EMAHelper
from scripts.repro_config import source_hashes
from scripts.repro_diagnostics import observed_update
from scripts.repro_eval import evaluate

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def set_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def on_device(x):
    import dataclasses
    if torch.is_tensor(x): return x.cuda()
    if dataclasses.is_dataclass(x):
        return type(x)(**{f.name: on_device(getattr(x, f.name)) for f in dataclasses.fields(x)})
    if isinstance(x, dict): return {k: on_device(v) for k, v in x.items()}
    if isinstance(x, list): return [on_device(v) for v in x]
    if isinstance(x, tuple): return tuple(on_device(v) for v in x)
    return x


def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


def trim_log(path, committed_step):
    """Remove uncommitted rows after a crash; a partial last line is discarded."""
    if not path.exists(): return
    tmp = path.with_suffix('.tmp')
    with path.open() as src, tmp.open('w') as dst:
        for line in src:
            try: row = json.loads(line)
            except json.JSONDecodeError: break
            if row['step'] <= committed_step: dst.write(line)
    os.replace(tmp, path)


def make_loader(config, split, rank, world, epochs):
    ds = PuzzleDataset(PuzzleDatasetConfig(seed=config.seed, dataset_paths=config.data_paths,
        global_batch_size=config.global_batch_size, test_set_mode=split == 'test',
        epochs_per_iter=epochs, rank=rank, num_replicas=world), split)
    # With no workers the loader base seed is unused. Isolate its consumption
    # from model RNG so reconstructing an iterator cannot perturb ACT sampling.
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(ds, batch_size=None, num_workers=0, pin_memory=True, generator=generator), ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seconds', type=int, default=160200)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--max-updates', type=int, default=0, help='Pause after this many additional updates; keep full recipe.')
    a = p.parse_args()
    if a.seconds <= 0 or a.checkpoint_every <= 0: p.error('Positive time and checkpoint interval required.')
    if 'LOCAL_RANK' not in os.environ: p.error('Launch with torchrun.')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    started = time.monotonic()  # Include initialization/loading in the walltime budget.
    rank = int(os.environ.get('RANK', '0')); world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    cpu_group = dist.new_group(backend='gloo')
    raw = json.loads(a.config.read_text()); config = upstream.PretrainConfig(**raw)
    diag = raw['diagnostics']
    if diag['every'] < 1 or config.epochs % config.eval_interval:
        raise ValueError('Invalid diagnostic cadence or native evaluation schedule.')
    hashes = source_hashes(ROOT)
    data_hashes = {p: hashlib.sha256((Path(p)/'DATA_READY.json').read_bytes()).hexdigest() for p in config.data_paths}
    fingerprint = hashlib.sha256(json.dumps(dict(config=raw, sources=hashes, data=data_hashes), sort_keys=True).encode()).hexdigest()
    out = Path(config.checkpoint_path); out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists(): raise RuntimeError('Run is already complete.')
    if not a.resume and any((out/name).exists() for name in ('resume.json', 'metrics.jsonl')):
        raise RuntimeError('Existing run: use --resume.')
    random.seed(config.seed+rank); np.random.seed(config.seed); torch.manual_seed(config.seed+rank)
    train_loader, train_ds = make_loader(config, 'train', rank, world, config.eval_interval)
    eval_loader, eval_ds = make_loader(config, 'test', rank, world, 1)
    state = upstream.init_train_state(config, train_ds.metadata, rank, world)
    ema = EMAHelper(config.ema_rate); ema.register(state.model)
    evaluators = upstream.create_evaluators(config, eval_ds.metadata)
    expected = 1 if raw['reproduction_dataset'] == 'arc2' else 0
    if len(evaluators) != expected: raise RuntimeError('Incorrect native evaluators.')
    iteration = 0; cursor = 0; dataset_before = 0; start_np = None
    phase = 'training'; evaluation = {}; restored = None; completed_evaluations = 0
    if a.resume:
        manifest = json.loads((out/'resume.json').read_text())
        if manifest['fingerprint'] != fingerprint or manifest['world_size'] != world:
            raise RuntimeError('Resume requires identical data, config, code and GPU count.')
        restored = torch.load(out/manifest['generation']/f'rank{rank}.pt', map_location='cpu', weights_only=False)
        state.model.load_state_dict(restored['model'])
        for opt, saved in zip(state.optimizers, restored['optimizers']): opt.load_state_dict(saved)
        state.carry = on_device(restored['carry']); state.step = restored['step']
        ema.load_state_dict(on_device(restored['ema']))
        iteration = restored['iteration']; cursor = restored['cursor']; dataset_before = restored['dataset_before']
        start_np = restored['start_numpy']; phase = restored['phase']; evaluation = on_device(restored['evaluation'])
        completed_evaluations = restored['completed_evaluations']
        for ev, cache in zip(evaluators, restored['evaluators']): ev._local_hmap, ev._local_preds = cache
        set_rng(restored['rng'])
        if rank == 0:
            for name in ('metrics.jsonl', 'diagnostics.jsonl'): trim_log(out/name, state.step)
            # Delete only uncommitted evaluation records, never checkpoints/data.
            for path in out.glob('evaluation-*.json'):
                if json.loads(path.read_text())['iteration'] >= completed_evaluations: path.unlink()
    initial_step = state.step
    if rank == 0:
        atomic_json(raw, out/'resolved_config.json')
        atomic_json(dict(source=hashes, data=data_hashes), out/'provenance.json')
        base = getattr(state.model, '_orig_mod', state.model); inner = base.model.inner
        dense = sum(p.numel() for p in base.parameters()); table = inner.puzzle_emb.weights.numel()
        atomic_json(dict(dense_learned=dense, puzzle_table_learned=table, total_learned=dense+table,
            reasoning_core=sum(p.numel() for p in inner.L_level.parameters()),
            layers=[sum(p.numel() for p in layer.parameters()) for layer in inner.L_level.layers]), out/'parameter_counts.json')
        atomic_json(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            world_size=world, reference_step_ceiling=state.total_steps, loader='native sampler, synchronous replay',
            numpy_seed=config.seed, compile_regular_updates=raw.get('study', {}).get('compile_mode','default') != 'eager',
            diagnostics=diag, source_commit=json.loads((ROOT/'repos.lock.json').read_text())['commit']), out/'runtime.json')

    def save():
        pointer = out/'resume.json'
        generation = 'resume-a' if not pointer.exists() or json.loads(pointer.read_text())['generation'] == 'resume-b' else 'resume-b'
        directory = out/generation; directory.mkdir(exist_ok=True)
        payload = dict(model=state.model.state_dict(), optimizers=[o.state_dict() for o in state.optimizers],
            carry=state.carry, step=state.step, ema=ema.state_dict(), rng=rng_state(), iteration=iteration,
            cursor=cursor, dataset_before=dataset_before, start_numpy=start_np, phase=phase,
            evaluation=evaluation, completed_evaluations=completed_evaluations,
            evaluators=[(e._local_hmap, e._local_preds) for e in evaluators])
        atomic_save(payload, directory/f'rank{rank}.pt'); dist.barrier()
        if rank == 0:
            atomic_json(dict(generation=generation, step=state.step, iteration=iteration, phase=phase,
                evaluation_batches=evaluation.get('cursor', 0), completed_evaluations=completed_evaluations,
                fingerprint=fingerprint, world_size=world), pointer)
        dist.barrier()

    def deadline():
        stop = STOP or (out/'STOP').exists() or time.monotonic()-started >= a.seconds
        flag = torch.tensor(int(stop), device='cuda'); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    total_iters = config.epochs//config.eval_interval
    while iteration < total_iters:
        if deadline():
            save(); dist.destroy_process_group(); return
        if phase == 'training':
            state.model.train()
            train_ds._iters = dataset_before
            if cursor:
                np.random.set_state(start_np)
            else:
                start_np = np.random.get_state()
            iterator = iter(train_loader)
            for _ in range(cursor): next(iterator)
            if cursor and restored is not None: set_rng(restored['rng'])
            restored = None
            for _, batch, effective_batch in iterator:
                t0 = time.monotonic(); next_step = state.step+1
                observed = next_step <= state.total_steps and (next_step <= diag['first_steps'] or next_step % diag['every'] == 0)
                accepted = (state.carry.halted.sum() if state.carry is not None else torch.tensor(batch['inputs'].shape[0], device='cuda')).clone()
                dist.all_reduce(accepted, op=dist.ReduceOp.SUM)
                if observed:
                    metrics, diagnostic = observed_update(upstream, config, state, batch, effective_batch, rank, world)
                else:
                    metrics = upstream.train_batch(config, state, batch, effective_batch, rank, world)
                # Match the upstream EMA cadence, including its ceiling behavior.
                ema.update(state.model); cursor += 1
                finite = metrics is None or all(np.isfinite(v) for v in metrics.values())
                flag = torch.tensor(int(finite), device='cuda'); dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if not flag.item(): raise RuntimeError('Non-finite training metric.')
                if rank == 0 and metrics is not None:
                    append(out/'metrics.jsonl', dict(step=state.step, iteration=iteration, seconds=time.monotonic()-t0,
                        diagnostic_update=observed, new_examples_accepted=int(accepted.item()),
                        **{k: float(v) for k, v in metrics.items()}))
                    if observed: append(out/'diagnostics.jsonl', dict(step=state.step, **diagnostic))
                    if state.step % 100 == 0 or state.step <= 10:
                        print(f"step={state.step}/{state.total_steps} lm_loss={metrics.get('train/lm_loss')} observed={observed}", flush=True)
                stop = deadline() or (a.max_updates > 0 and state.step-initial_step >= a.max_updates)
                if stop or state.step % a.checkpoint_every == 0: save()
                if stop:
                    dist.destroy_process_group(); return
            dataset_before = train_ds._iters
            phase = 'evaluation'; evaluation = {}; cursor = 0; start_np = None
            save()
        if phase == 'evaluation':
            if deadline():
                save(); dist.destroy_process_group(); return
            eval_model = ema.ema_copy(state.model); eval_model.eval()
            cfg = SimpleNamespace(checkpoint_path=config.checkpoint_path, eval_save_outputs=[], step=state.step)
            done, metrics = evaluate(cfg, eval_model, eval_loader, eval_ds.metadata, evaluators,
                rank, world, cpu_group, evaluation, save, deadline)
            if not done:
                dist.destroy_process_group(); return
            if rank == 0:
                atomic_json(dict(step=state.step, iteration=iteration, metrics=metrics), out/f'evaluation-{iteration:02d}.json')
                atomic_save(eval_model.state_dict(), out/f'step_{state.step}')
            del eval_model
            dist.barrier()
            completed_evaluations += 1; iteration += 1
            phase = 'training'; evaluation = {}; cursor = 0; start_np = None
            save()
    if rank == 0:
        atomic_json(dict(step=state.step, optimizer_updates=min(state.step, state.total_steps),
            completed_evaluations=completed_evaluations, fingerprint=fingerprint,
            status='configured_schedule_and_evaluations_completed', seed=config.seed), out/'COMPLETE.json')
    dist.destroy_process_group()


if __name__ == '__main__': main()

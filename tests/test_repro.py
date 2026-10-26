"""CPU correctness checks. AdamW is a TEST shim, never a production fallback."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from study import bootstrap
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as Native
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from tests.test_design import configuration, batch
from scripts.repro_diagnostics import UpdateObserver
from scripts.repro_config import configuration as recipe
shim = types.ModuleType('adam_atan2'); shim.AdamATan2 = torch.optim.AdamW
with patch.dict(sys.modules, {'adam_atan2': shim}):
    from scripts import repro_train as driver
from scripts.repro_eval import evaluate


class DiagnosticsTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_hooks_preserve_native_loss_gradients_updates_and_rng(self):
        torch.manual_seed(7)
        a = ACTLossHead(Native(configuration()), 'stablemax_cross_entropy'); b = copy.deepcopy(a)
        x = batch(); oa = torch.optim.AdamW(a.parameters(), lr=1e-4); ob = torch.optim.AdamW(b.parameters(), lr=1e-4)
        old = [[p.detach().clone() for p in layer.parameters()] for layer in b.model.inner.L_level.layers]
        torch.manual_seed(42)
        ca, la, *_ = a(carry=a.initial_carry(x), batch=x, return_keys=[])
        (la/2).backward(); oa.step(); rng = torch.get_rng_state()
        torch.manual_seed(42)
        outputs = []
        hook = b.model.inner.L_level.register_forward_hook(lambda m, args, out: outputs.append(out))
        with UpdateObserver(b, ob, 2) as observer:
            cb, lb, *_ = b(carry=b.initial_carry(x), batch=x, return_keys=[])
            gz, gy = torch.autograd.grad(lb, outputs[-2:], retain_graph=True)
            # autograd.grad above also invokes hooks, but the true native
            # backward overwrites the observations with its scaled gradients.
            (lb/2).backward(); ob.step()
        hook.remove(); stats = observer.finish()
        torch.testing.assert_close(la, lb, atol=0, rtol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, atol=0, rtol=0)
            torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)
        torch.testing.assert_close(ca.inner_carry.z_L, cb.inner_carry.z_L, atol=0, rtol=0)
        for name, grad in [('z', gz), ('y', gy)]:
            self.assertAlmostEqual(stats['latents'][name]['gradient']['l2'], grad.double().norm().item(), places=7)
        for i, layer in enumerate(b.model.inner.L_level.layers):
            expected = math.sqrt(sum((p.detach().double()-q.double()).square().sum().item() for p, q in zip(layer.parameters(), old[i])))
            self.assertAlmostEqual(stats['layers'][f'layer_{i+1}']['update_l2'], expected, places=7)
        self.assertFalse(b.model.inner.L_level._forward_hooks)

    def test_recipe_matches_vendored_hydra_defaults_and_readme_overrides(self):
        from omegaconf import OmegaConf
        native = OmegaConf.load(bootstrap.VENDOR/'config/cfg_pretrain.yaml')
        arch = OmegaConf.to_container(OmegaConf.load(bootstrap.VENDOR/'config/arch/trm.yaml'), resolve=True)
        for dataset in ('arc2', 'sudoku'):
            cfg = recipe(dataset)
            for key in ('global_batch_size', 'lr', 'lr_min_ratio', 'lr_warmup_steps', 'beta1', 'beta2', 'seed', 'ema_rate'):
                self.assertEqual(cfg[key], native[key])
            for key in arch:
                self.assertEqual(cfg['arch'][key], (4 if dataset == 'arc2' else 6) if key == 'L_cycles' else arch[key])
            self.assertTrue(cfg['ema']); self.assertEqual(cfg['epochs']//cfg['eval_interval'], 10)
        self.assertEqual(recipe('sudoku')['weight_decay'], 1.0)
        self.assertEqual(recipe('sudoku')['puzzle_emb_lr'], 1e-4)
        self.assertEqual(recipe('arc2')['epochs'], 100000)


class Voter:
    required_outputs = {'preds'}
    def __init__(self): self._local_hmap = {}; self._local_preds = {}
    def begin_eval(self): pass  # Match native aggregated_voting=True across checkpoints.
    def update_batch(self, batch, preds):
        for key in batch['puzzle_identifiers'].tolist(): self._local_preds.setdefault(key, []).append(1)
    def result(self, *args, **kwargs): return {'test_votes': sum(map(len, self._local_preds.values()))}


class DriverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.data = self.root/'data'; self.data.mkdir()
        (self.data/'DATA_READY.json').write_text('{}')
        for split in ('train', 'test'):
            p = self.data/split; p.mkdir()
            metadata = dict(seq_len=9, vocab_size=12, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                num_puzzle_identifiers=5, total_groups=4, total_puzzles=4, mean_puzzle_examples=1, sets=['all'])
            (p/'dataset.json').write_text(json.dumps(metadata))
            arrays = dict(inputs=np.full((12, 9), 3), labels=np.full((12, 9), 4),
                puzzle_identifiers=np.repeat(np.arange(1, 5), 3), puzzle_indices=np.arange(13),
                group_indices=np.arange(0, 13, 3))
            for k, v in arrays.items(): np.save(p/f'all__{k}.npy', v.astype(np.int32))

    def tearDown(self): self.temp.cleanup()

    def config(self, name):
        cfg = recipe('arc2', self.root)
        cfg.update(data_paths=[str(self.data)], epochs=2, eval_interval=1, global_batch_size=2,
                   lr_warmup_steps=2, checkpoint_path=str(self.root/name))
        cfg['diagnostics'].update(every=2, first_steps=1)
        path = self.root/(name+'.json'); path.write_text(json.dumps(cfg)); return path

    def run_driver(self, path, resume=False, limit=0, interrupt_eval=False):
        real_loader = driver.make_loader
        def loader(*args):
            result, ds = real_loader(*args)
            result.pin_memory = False
            return result, ds
        def init(cfg, metadata, rank, world):
            model = ACTLossHead(Native(configuration()), 'stablemax_cross_entropy')
            optim = [CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), world_size=1),
                     torch.optim.AdamW(model.parameters(), lr=0, betas=(.9, .95), weight_decay=.1)]
            return driver.upstream.TrainState(model, optim, [.01, .0001], None, 0, 4)
        def ev(*args, **kwargs):
            args = list(args)
            if interrupt_eval: args[-1] = lambda: True
            return evaluate(*args, device='cpu', checkpoint_every=2, **kwargs)
        argv = ['train', '--config', str(path), '--max-updates', str(limit)]
        if resume: argv.append('--resume')
        device = torch.device; tensor = torch.tensor
        def cpu_tensor(*args, **kwargs): kwargs['device'] = 'cpu'; return tensor(*args, **kwargs)
        with ExitStack() as stack:
            patches = [patch.object(driver.upstream, 'init_train_state', init),
                patch.object(driver, 'make_loader', loader),
                patch.object(driver.upstream, 'create_evaluators', lambda *args: [Voter()]),
                patch.object(driver, 'evaluate', ev), patch.object(driver, 'STOP', False),
                patch.object(torch, 'tensor', cpu_tensor), patch.object(torch, 'device', lambda *args, **kw: device('cpu')),
                patch.object(torch.Tensor, 'cuda', lambda self, *args, **kw: self), patch.object(driver.signal, 'signal'),
                patch.dict(os.environ, {'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1'}), patch.object(sys, 'argv', argv),
                patch.object(torch.cuda, 'get_rng_state', torch.get_rng_state), patch.object(torch.cuda, 'set_rng_state', torch.set_rng_state),
                patch.object(torch.cuda, 'get_device_name', lambda: 'CPU TEST ONLY'), patch.object(torch.cuda, 'set_device')]
            for name in ('init_process_group', 'new_group', 'all_reduce', 'barrier', 'destroy_process_group'):
                patches.append(patch.object(driver.dist, name))
            for obj in patches: stack.enter_context(obj)
            driver.main()

    def saved(self, name):
        out = self.root/name; meta = json.loads((out/'resume.json').read_text())
        return torch.load(out/meta['generation']/'rank0.pt', weights_only=False)

    def compare(self, a, b):
        sa, sb = self.saved(a), self.saved(b)
        for key in ('model', 'ema', 'optimizers'): torch.testing.assert_close(sa[key], sb[key], atol=0, rtol=0)
        self.assertEqual(sa['step'], 4); self.assertEqual(sb['completed_evaluations'], 2)
        for iteration in (0, 1):
            ra = json.loads((self.root/a/f'evaluation-{iteration:02d}.json').read_text())
            rb = json.loads((self.root/b/f'evaluation-{iteration:02d}.json').read_text())
            self.assertEqual(ra, rb)
            self.assertEqual(rb['metrics']['test_votes'], 12*(iteration+1))

    def test_training_resume_and_crash_log_truncation(self):
        a = self.config('continuous'); self.run_driver(a)
        b = self.config('resumed'); self.run_driver(b, limit=1)
        out = self.root/'resumed'
        with (out/'metrics.jsonl').open('a') as f: f.write('{"step": 999, "uncommitted": true}\n{"step":')
        self.run_driver(b, resume=True)
        self.compare('continuous', 'resumed')
        self.assertEqual([json.loads(x)['step'] for x in (out/'metrics.jsonl').read_text().splitlines()], [1, 2, 3, 4])

    def test_evaluation_resume_preserves_all_scheduled_votes(self):
        a = self.config('continuous'); self.run_driver(a)
        b = self.config('resumed'); self.run_driver(b, interrupt_eval=True)
        self.assertEqual(self.saved('resumed')['phase'], 'evaluation')
        self.assertEqual(self.saved('resumed')['evaluation']['cursor'], 1)
        self.run_driver(b, resume=True)
        self.compare('continuous', 'resumed')


if __name__ == '__main__': unittest.main()

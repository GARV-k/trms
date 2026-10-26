"""Resolve the pinned upstream attention recipes without importing torch."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]


def source_hashes(root=ROOT):
    files = [root/'study/bootstrap.py', root/'repos.lock.json']
    files += sorted((root/'scripts').glob('repro_*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def configuration(dataset, root=ROOT):
    if dataset not in ('arc2', 'sudoku'):
        raise ValueError('Choose arc2 or sudoku explicitly.')
    sudoku = dataset == 'sudoku'
    name = f'repro-attention-{dataset}-seed0'
    folder = 'sudoku-extreme-1k-aug-1000' if sudoku else 'arc2concept-aug-1000'
    return dict(
        arch=dict(name='recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1',
            loss=dict(name='losses@ACTLossHead', loss_type='stablemax_cross_entropy'),
            H_cycles=3, L_cycles=6 if sudoku else 4, H_layers=0, L_layers=2,
            hidden_size=512, num_heads=8, expansion=4, puzzle_emb_ndim=512,
            puzzle_emb_len=16, pos_encodings='rope', forward_dtype='bfloat16',
            mlp_t=False, no_ACT_continue=True, halt_max_steps=16, halt_exploration_prob=0.1),
        data_paths=[str(root/'data/paper'/folder)], data_paths_test=[],
        evaluators=[] if sudoku else [dict(name='arc@ARC', aggregated_voting=True)],
        global_batch_size=768, epochs=50000 if sudoku else 100000,
        eval_interval=5000 if sudoku else 10000, min_eval_interval=0,
        checkpoint_every_eval=True, eval_save_outputs=[],
        lr=1e-4, lr_min_ratio=1.0, lr_warmup_steps=2000, beta1=0.9, beta2=0.95,
        weight_decay=1.0 if sudoku else 0.1, puzzle_emb_weight_decay=1.0 if sudoku else 0.1,
        puzzle_emb_lr=1e-4 if sudoku else 0.01, seed=0, ema=True, ema_rate=0.999,
        freeze_weights=False, project_name='TRM-Reproduction', run_name=name,
        checkpoint_path=str(root/'results'/name),
        diagnostics=dict(every=100, first_steps=10, mode='eager_observed_updates'),
        reproduction_dataset=dataset)


def prepare(dataset, root=ROOT):
    config = configuration(dataset, root)
    path = root/'configs/resolved'/(config['run_name']+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, indent=2)+'\n'
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f'Refusing to replace a changed configuration: {path}')
    path.write_text(text)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['arc2', 'sudoku'], required=True)
    print(prepare(parser.parse_args().dataset))

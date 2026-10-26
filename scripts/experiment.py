"""Explicit three-axis design; no selection using the held-out test set."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from scripts.repro_config import configuration as original

ROOT = Path(__file__).absolute().parents[1]


def specs(axis2_k=3):
    if axis2_k < 1: raise ValueError('Positive paper-table k required.')
    cases = {'preliminary': dict(k=1,t=3,arm='native',stage='preliminary')}
    for k in (1,2,4,6): cases[f'axis1-k{k}'] = dict(k=k,t=3,arm='native',stage='axis1')
    for t in (1,2,3,4): cases[f'axis2-t{t}'] = dict(k=axis2_k,t=t,arm='native',stage='axis2')
    for k in (1,2,4,6):
        for arm in ('tied_full','untied_full'):
            cases[f'axis3-k{k}-{arm}'] = dict(k=k,t=3,arm=arm,stage='axis3')
    return cases


def make_config(k=1, t=3, arm='native', batch=768, epochs=50000, width=256, root=ROOT,
                mode='reduce-overhead', started=None):
    if k < 1 or t < 1: raise ValueError('k and t must be positive integers; native N=2*k.')
    if arm not in ('native','tied_full','untied_full'): raise ValueError('Unknown model arm.')
    if batch < 4 or batch % 4: raise ValueError('Batch must be divisible by four.')
    if width < 16 or width % 16: raise ValueError('Width must be divisible by 16 (8 heads, even RoPE head dimension).')
    if epochs < 10 or epochs % 10: raise ValueError('Epochs must be a positive multiple of ten.')
    if mode not in ('reduce-overhead','default','eager'): raise ValueError('Unsupported compilation mode.')
    cfg = original('sudoku',root)
    cfg['arch'].update(hidden_size=width,puzzle_emb_ndim=width,L_cycles=2*k,H_cycles=t)
    if arm!='native': cfg['arch'].update(name='sudoku_controls@Model',untied=arm=='untied_full')
    cfg.update(global_batch_size=batch,epochs=epochs,eval_interval=epochs//10,
        data_paths=[str(root/'data/paper/sudoku-extreme-1k-aug-800')])
    started = started or datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    identity = f"{started}-sudoku-d{width}-k{k}-t{t}-{arm}-b{batch}-e{epochs}-{mode}-s0"
    cfg.update(run_name=identity,checkpoint_path=str(root/'results'/identity),
        study=dict(paper_table_k=k,native_N=2*k,T=t,
            attention_blocks_per_supervision=2*t*(2*k+1),arm=arm,
            width=width,augmentation=800,world_size=4,compile_mode=mode,
            gradient_horizon='last_T_cycle' if arm=='native' else 'all_T_cycles',
            exact_paper_reproduction=False))
    return cfg


def configuration(case, batch=768, epochs=50000, axis2_k=3, root=ROOT, mode='reduce-overhead'):
    spec=specs(axis2_k)[case]
    return make_config(spec['k'],spec['t'],spec['arm'],batch,epochs,root=root,mode=mode)


def save_config(cfg):
    path=ROOT/'configs/resolved'/(cfg['run_name']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(cfg,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:raise RuntimeError('Different existing config: '+str(path))
    path.write_text(text)
    return path


def source_hashes(root=ROOT):
    paths=[root/'repos.lock.json']+sorted((root/'scripts').glob('*.py'))+sorted((root/'study').glob('*.py'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def resolved(case):
    plan=json.loads((ROOT/'configs/plan.json').read_text())
    cfg=configuration(case,plan['batch'],plan['epochs'],plan['axis2_k'],mode=plan['compile_mode'])
    path=ROOT/'configs/resolved'/(cfg['run_name']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(cfg,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:raise RuntimeError('Different existing config: '+str(path))
    path.write_text(text)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=list(specs()));a=p.parse_args()
    print(resolved(a.case))

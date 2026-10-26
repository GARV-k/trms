"""Render exactly three scientific figures from committed real training records."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'trm-matplotlib'))


def rows(path, ceiling):
    if not path.exists(): return []
    result = {}
    for line in path.read_text().splitlines():
        try: value = json.loads(line)
        except json.JSONDecodeError: continue
        if value['step'] <= ceiling: result[value['step']] = value
    return [result[k] for k in sorted(result)]


def flatten(row, prefix=''):
    result = {}
    for key, value in row.items():
        name = prefix+key
        if isinstance(value, dict): result.update(flatten(value, name+'.'))
        else: result[name] = value
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--smooth', type=int, default=200, help='Trailing mean window in optimizer updates.')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.smooth < 1: p.error('--smooth must be positive')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = a.run
    commit = json.loads((run/'resume.json').read_text())
    metrics = rows(run/'metrics.jsonl', commit['step'])
    diagnostics = rows(run/'diagnostics.jsonl', commit['step'])
    if not metrics or not diagnostics:
        raise SystemExit('Need at least one committed training and diagnostic update.')
    out = a.out or run/'plots'; out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    status = 'complete schedule' if (run/'COMPLETE.json').exists() else 'partial run'
    cfg = json.loads((run/'resolved_config.json').read_text())
    title = f"TRM attention · {cfg['reproduction_dataset']} · seed {cfg['seed']} · {status}"
    def save(fig, name):
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        for suffix in ('png', 'pdf'): fig.savefig(out/(name+'.'+suffix), bbox_inches='tight')
        plt.close(fig)
    def smooth(values):
        values = np.asarray(values, dtype=float)
        sums = np.concatenate(([0.], np.cumsum(values)))
        end = np.arange(1, len(values)+1); start = np.maximum(0, end-a.smooth)
        return (sums[end]-sums[start])/(end-start)
    def style(ax, ylabel):
        ax.set_xlabel('Optimizer updates'); ax.set_ylabel(ylabel)
        ax.grid(alpha=.2); ax.legend(frameon=False)
    def norm_axis(ax):
        values = np.concatenate([line.get_ydata() for line in ax.lines])
        if np.all(values > 0) and values.max()/values.min() >= 100:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    step = [r['step'] for r in metrics]
    lm = [r['train/lm_loss'] for r in metrics]
    total = [r['train/lm_loss']+.5*r['train/q_halt_loss'] for r in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stride = max(1, len(step)//5000)
    ax.plot(step[::stride], np.asarray(lm)[::stride], color='#3366aa', alpha=.15, linewidth=.5)
    ax.plot(step, smooth(lm), label=f'Train token-mean LM loss ({a.smooth}-update mean)', color='#3366aa')
    ax.plot(step, smooth(total), label='Train objective: LM + 0.5 × halt', color='#bb6633')
    evaluations = []
    for path in sorted(run.glob('evaluation-*.json')):
        value = json.loads(path.read_text())
        if value['iteration'] < commit['completed_evaluations']: evaluations.append(value)
    eval_points = [(v['step'], v['metrics']['all']['lm_loss']) for v in evaluations
                   if 'all' in v['metrics'] and 'lm_loss' in v['metrics']['all']]
    if eval_points:
        x, y = zip(*eval_points); ax.plot(x, y, 'o--', label='EMA evaluation LM loss', color='#227755')
    style(ax, 'Loss'); save(fig, '01_loss')
    step = [r['step'] for r in diagnostics]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for state, color in [('z', '#3366aa'), ('y', '#bb6633')]:
        axes[0].plot(step, [r['latents'][state]['gradient']['rms'] for r in diagnostics], label=state, color=color)
        axes[1].plot(step, [r['latents'][state]['forward_change']['rms'] for r in diagnostics], label=state, color=color)
    style(axes[0], 'Final-state gradient RMS (per-example loss)')
    style(axes[1], 'State change RMS over one supervision step')
    norm_axis(axes[0])
    save(fig, '02_latent_norms')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names=sorted(diagnostics[0]['layers'],key=lambda s:int(s.split('_')[-1]))
    for position, color in [(0, '#3366aa'), (1, '#bb6633')]:
        group=names[position::2]
        label=f'Layer {position+1}' if len(group)==1 else f'Block position {position+1}: median / range across {len(group)} copies'
        for ax,key in zip(axes,('grad_rms','relative_update')):
            values=np.asarray([[r['layers'][name][key] for name in group] for r in diagnostics])
            ax.plot(step,np.median(values,axis=1),label=label,color=color)
            if len(group)>1:ax.fill_between(step,values.min(axis=1),values.max(axis=1),color=color,alpha=.15)
    style(axes[0], 'Layer gradient RMS (global mean objective)')
    style(axes[1], 'Actual weight update L2 / weight L2')
    for ax in axes: norm_axis(ax)
    save(fig, '03_layer_norms')
    flat = [flatten(r) for r in diagnostics]
    with (out/'diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    inputs = ['metrics.jsonl', 'diagnostics.jsonl', 'resolved_config.json', 'resume.json']
    inputs += [p.name for p in sorted(run.glob('evaluation-*.json'))]
    (out/'plot_manifest.json').write_text(json.dumps(dict(committed_step=commit['step'], status=status,
        smoothing_updates=a.smooth, sources={n: hashlib.sha256((run/n).read_bytes()).hexdigest() for n in inputs},
        figures=['01_loss', '02_latent_norms', '03_layer_norms']), indent=2)+'\n')
    print(out)


if __name__ == '__main__': main()

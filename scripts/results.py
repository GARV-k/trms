"""Print completed and partial Sudoku results without selecting a winner."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).absolute().parents[1]/'results')
    a=p.parse_args()
    for config in sorted(a.root.glob('*/resolved_config.json')):
        cfg=json.loads(config.read_text());run=config.parent
        committed=json.loads((run/'resume.json').read_text()) if (run/'resume.json').exists() else None
        if committed is None:continue
        events=[json.loads(f.read_text()) for f in sorted(run.glob('evaluation-*.json'))]
        # Ignore evaluations written after the last atomic checkpoint pointer.
        events=[e for e in events if e['step']<=committed['step'] and
                e['iteration']<committed['iteration']]
        print(run.name)
        print('  status:', 'COMPLETE' if (run/'COMPLETE.json').exists() else 'PARTIAL',
              '| committed optimizer step:',committed['step'])
        if events:
            last=events[-1];m=last['metrics']['all']
            print(f"  final available test exact accuracy: {100*m['exact_accuracy']:.2f}% | LM loss: {m['lm_loss']:.5f} | evaluation {last['iteration']+1}/10")
        else:print('  no committed evaluation yet')
        print('  k/N/T:',cfg['study']['paper_table_k'],cfg['study']['native_N'],cfg['study']['T'],
              '| width:',cfg['arch']['hidden_size'],'| batch:',cfg['global_batch_size'],'| epochs:',cfg['epochs'])


if __name__=='__main__':main()

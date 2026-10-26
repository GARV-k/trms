"""Launch exactly one experiment from explicit axis arguments."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment import make_config, save_config


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis',choices=['preliminary','1','2','3'],required=True)
    p.add_argument('--k',type=int,help='Paper-table k; native TRM N=2*k. Default: 1, or 3 for axis 2.')
    p.add_argument('--t',type=int,default=3,help='Native H_cycles T (default 3).')
    p.add_argument('--arm',choices=['native','tied_full','untied_full'],default='native')
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--batch',type=int,help='Global batch; default from the common profile plan.')
    p.add_argument('--epochs',type=int,help='Native epochs, divisible by 10; default from common profile plan.')
    p.add_argument('--mode',choices=['reduce-overhead','default','eager'],help='Default from profile plan, otherwise reduce-overhead.')
    p.add_argument('--plan',type=Path,default=ROOT/'configs/plan.json')
    p.add_argument('--max-updates',type=int,default=0,help='Pause after this many additional updates; 0 runs the full planned schedule.')
    p.add_argument('--dry-run',action='store_true',help='Print config without writing or training.')
    return p


def resolve(args):
    plan=json.loads(args.plan.read_text()) if args.plan.exists() else {}
    batch=args.batch if args.batch is not None else plan.get('batch')
    epochs=args.epochs if args.epochs is not None else plan.get('epochs')
    if batch is None or epochs is None: raise ValueError('Run the profile job first, or provide both --batch and --epochs explicitly.')
    k=args.k if args.k is not None else (plan.get('axis2_k',3) if args.axis=='2' else 1)
    if args.axis=='3' and args.arm=='native':raise ValueError('Axis 3 requires --arm tied_full or --arm untied_full.')
    if args.axis!='3' and args.arm!='native':raise ValueError('Use --axis 3 for full-gradient tied/untied controls.')
    if args.axis in ('preliminary','1') and args.t!=3:raise ValueError('Preliminary / axis 1 fix T=3; use axis 2 to vary T.')
    return make_config(k,args.t,args.arm,batch,epochs,args.width,mode=args.mode or plan.get('compile_mode','reduce-overhead'))


def main():
    p=parser();args=p.parse_args()
    try: cfg=resolve(args)
    except ValueError as e:p.error(str(e))
    if args.dry_run:print(json.dumps(cfg,indent=2));return 0
    if int(os.environ.get('SLURM_JOB_NUM_NODES','0'))!=1: p.error('Submit slurm/train.sbatch on one four-GH200 node.')
    path=save_config(cfg)
    remaining=int(os.environ['SLURM_JOB_END_TIME'])-int(time.time())-1800
    if remaining<900:raise RuntimeError('Insufficient time for training and a safe checkpoint.')
    print('One experiment:',cfg['run_name'], '\nResolved configuration:',path,flush=True)
    result=subprocess.run([sys.executable,'scripts/run.py','--config',str(path),
        '--seconds',str(remaining),'--max-updates',str(args.max_updates)],cwd=ROOT)
    if result.returncode==0:
        subprocess.run(['bash','scripts/container.sh','scripts/repro_plot.py','--run',cfg['checkpoint_path']],cwd=ROOT,check=True)
        return 0
    if result.returncode in (3,4):
        print('Paused safely. Resubmit the SAME arguments to resume. This run is not complete.',flush=True)
        return 4
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

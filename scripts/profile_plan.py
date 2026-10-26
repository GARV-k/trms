"""Select a common safe batch and epoch budget before running individual cases."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.processes import command, run_case
from scripts.experiment import source_hashes


def safe(record):
    return (record.get('status')=='passed' and
        0 < record.get('peak_memory_fraction',math.inf) <= .8 and
        0 < record.get('projected_compute_hours',math.inf) < math.inf)


def budget_epochs(records, batch, training_target_hours=10):
    # Reserve two hours of the 12-hour allocation for startup, checkpoints and
    # estimation error; add a further 25% allowance to measured GPU computation.
    allowed=training_target_hours/1.25
    maximum=50000
    for r in records:
        eval_h=r['projected_evaluation_hours']
        if eval_h>=allowed: raise RuntimeError('Ten evaluations alone exceed the budget. No plan was created.')
        n=math.floor(50000*(allowed-eval_h)/r['projected_training_hours']/10)*10
        maximum=min(maximum,n)
    if maximum < 10 or maximum*1000//batch < 4000:
        raise RuntimeError('Budget leaves fewer than 4,000 updates (2,000 warmup). Lower batch or increase time; no plan created.')
    return maximum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis2-k',type=int,default=3)
    p.add_argument('--max-batch',type=int,default=8192)
    p.add_argument('--mode',choices=['default','reduce-overhead'],default='reduce-overhead')
    p.add_argument('--out',type=Path,default=ROOT/'results/profile'/('job-'+os.environ.get('SLURM_JOB_ID','local')))
    a=p.parse_args()
    plan_path=ROOT/'configs/plan.json'
    if plan_path.exists():p.error('A common plan already exists; reuse it to keep axis comparisons consistent.')
    if a.axis2_k<1 or a.max_batch<192:p.error('Positive k and max batch >=192 required.')
    a.out.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,'scripts/verify_vendor.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/repro_data.py','audit','--dataset','sudoku'],cwd=ROOT,check=True)
    if run_case(command('scripts.preflight',[]),a.out/'preflight.log',900):raise RuntimeError('Preflight failed; inspect its log.')
    history=[]
    def probe(case,batch):
        out=a.out/f'{case}-b{batch}.json';log=out.with_suffix('.log')
        remaining=int(os.environ.get('SLURM_JOB_END_TIME',str(int(time.time())+7200)))-int(time.time())-300
        if remaining<300:raise RuntimeError('Profile allocation nearly exhausted; no training submitted.')
        print(f'Profiling {case}, batch {batch}; {log}',flush=True)
        status=run_case(command('scripts.profile_case',['--case',case,'--batch',str(batch),
            '--axis2-k',str(a.axis2_k),'--mode',a.mode,'--out',str(out.absolute())]),log,min(3600,remaining))
        r=json.loads(out.read_text()) if status==0 and out.exists() else dict(status='failed',global_batch_size=batch,exit_code=status,log=str(log))
        r['case']=case;history.append(r)
        (a.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        return r
    candidates=[]
    for batch in (384,768,1536,3072,6144,8192):
        if batch>a.max_batch:continue
        r=probe('axis3-k6-untied_full',batch)
        if safe(r):candidates.append(r)
        else: break  # Stop increasing after insufficient headroom or failure.
    if not candidates:
        r=probe('axis3-k6-untied_full',192)
        if safe(r):candidates.append(r)
    # Fastest tested complete computation, not simply largest allocation.
    chosen=None
    for candidate in sorted(candidates,key=lambda r:r['projected_compute_hours']):
        batch=candidate['global_batch_size']
        records=[candidate]+[probe(case,batch) for case in
            ('axis1-k6','axis2-t4','axis3-k6-tied_full','preliminary')]
        if not all(safe(r) for r in records):continue
        try:epochs=budget_epochs(records,batch)
        except RuntimeError as e:print(str(e),flush=True);continue
        chosen=(batch,epochs,records);break
    if chosen is None:raise RuntimeError('No common memory/time-safe plan found; inspect profile logs. No training submitted.')
    batch,epochs,records=chosen
    projections={r['case']:1.25*(r['projected_training_hours']*epochs/50000+r['projected_evaluation_hours']) for r in records}
    plan=dict(batch=batch,epochs=epochs,eval_interval=epochs//10,axis2_k=a.axis2_k,compile_mode=a.mode,
        width=256,augmentation=800,original_epochs=50000,original_batch=768,
        estimated_hours_with_margin=projections,profile_directory=str(a.out),sources=source_hashes(),
        note='Fixed common batch and epoch budget for all three axes. Native optimizer/LR unchanged; larger batch changes update count. Estimates exclude queue time and are not a deadline guarantee.')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2),flush=True)
    print('Plan saved. No training was submitted. Run one experiment with slurm/train.sbatch.',flush=True)


if __name__=='__main__':main()

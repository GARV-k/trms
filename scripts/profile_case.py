"""Real four-GPU training/EMA-evaluation probe, discarded before final training."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import random
from study import bootstrap
import numpy as np
import torch
import torch.distributed as dist
import pretrain as upstream
from models.ema import EMAHelper
from scripts.repro_train import make_loader
from scripts.repro_diagnostics import observed_update
from scripts.single4_compile import ManagedCompile
from scripts.experiment import configuration, source_hashes, specs


def maximum(value):
    x=torch.tensor(float(value),device='cuda');dist.all_reduce(x,op=dist.ReduceOp.MAX)
    return x.item()


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--case',choices=list(specs()),required=True);p.add_argument('--axis2-k',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['reduce-overhead','default'],default='reduce-overhead')
    p.add_argument('--updates',type=int,default=40);p.add_argument('--warmup',type=int,default=20)
    args=p.parse_args()
    if args.updates<10 or args.warmup<10:p.error('Use >=10 warmup and timed updates.')
    if int(os.environ.get('WORLD_SIZE','0'))!=4:raise RuntimeError('Profile on exactly four GPUs.')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl');rank=dist.get_rank()
    dist.new_group(backend='gloo')
    raw=configuration(args.case,batch=args.batch,axis2_k=args.axis2_k,mode=args.mode)
    config=upstream.PretrainConfig(**raw)
    torch.manual_seed(config.seed+rank);np.random.seed(config.seed);random.seed(config.seed+rank)
    loader,ds=make_loader(config,'train',rank,4,config.eval_interval)
    eval_loader,eval_ds=make_loader(config,'test',rank,4,1)
    factory=upstream.create_model
    def create(*a,**kw):
        prior=os.environ.get('DISABLE_COMPILE');os.environ['DISABLE_COMPILE']='1'
        try:model,optimizers,lrs=factory(*a,**kw)
        finally:
            if prior is None:os.environ.pop('DISABLE_COMPILE',None)
            else:os.environ['DISABLE_COMPILE']=prior
        return ManagedCompile(model,args.mode),optimizers,lrs
    upstream.create_model=create
    state=upstream.init_train_state(config,ds.metadata,rank,4)
    ema=EMAHelper(config.ema_rate);ema.register(state.model)
    total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    free,_=torch.cuda.mem_get_info()
    non_torch=max(0,total-free-torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    state.model.train();iterator=iter(loader);times=[]
    for i in range(args.warmup+args.updates):
        _,batch,gb=next(iterator)
        torch.cuda.synchronize();start=time.perf_counter()
        metrics=upstream.train_batch(config,state,batch,gb,rank,4);ema.update(state.model)
        torch.cuda.synchronize();elapsed=maximum(time.perf_counter()-start)
        finite=int(metrics is None or all(np.isfinite(v) for v in metrics.values()))
        flag=torch.tensor(finite,device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if not flag.item():raise RuntimeError('Non-finite training metric.')
        if i>=args.warmup:times.append(elapsed)
    _,batch,gb=next(iterator)
    torch.cuda.synchronize();start=time.perf_counter()
    metrics,diagnostics=observed_update(upstream,config,state,batch,gb,rank,4);ema.update(state.model)
    torch.cuda.synchronize();diagnostic_seconds=maximum(time.perf_counter()-start)
    # Keep training model, optimizer state and EMA resident, as during real evaluation.
    model=ema.ema_copy(state.model);model.eval()
    _,eval_batch,_=next(iter(eval_loader));eval_batch={k:v.cuda() for k,v in eval_batch.items()}
    inference=[]
    with torch.inference_mode():
        for i in range(4):
            with torch.device('cuda'):carry=model.initial_carry(eval_batch)
            torch.cuda.synchronize();start=time.perf_counter();steps=0
            while True:
                carry,loss,metrics,preds,done=model(carry=carry,batch=eval_batch,return_keys={'preds'})
                steps+=1
                if done:break
            torch.cuda.synchronize();elapsed=maximum(time.perf_counter()-start)
            if steps!=16:raise RuntimeError('Expected full 16-step native evaluation.')
            finite=torch.stack([torch.isfinite(loss).all()]+[torch.isfinite(v).all() for v in metrics.values()]).all().int()
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():raise RuntimeError('Non-finite evaluation metric.')
            if i:inference.append(elapsed)
            del carry,loss,metrics,preds
    # CUDA private graph pools are included in allocator reserved memory.
    peak=maximum(torch.cuda.max_memory_reserved()+non_torch)
    fraction=maximum((torch.cuda.max_memory_reserved()+non_torch)/total)
    allocated=maximum(torch.cuda.max_memory_allocated())
    eval_sizes=[len(np.load(Path(raw['data_paths'][0])/'test'/(name+'__inputs.npy'),mmap_mode='r')) for name in eval_ds.metadata.sets]
    eval_examples=sum(eval_sizes)
    updates=state.total_steps
    update_seconds=statistics.median(times);eval_seconds=statistics.median(inference)
    diag_cost=(10+updates//100)*max(0,diagnostic_seconds-update_seconds)
    training_hours=(updates*update_seconds+diag_cost)/3600
    evaluation_hours=10*sum(math.ceil(size/args.batch) for size in eval_sizes)*eval_seconds/3600
    if rank==0:
        value=dict(status='passed',global_batch_size=args.batch,batch_per_gpu=args.batch//4,
            world_size=4,compile_mode=args.mode,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            training_update_seconds=times,median_update_seconds=update_seconds,
            training_examples_per_second=args.batch/update_seconds,
            diagnostic_update_seconds=diagnostic_seconds,evaluation_batch_seconds=inference,
            peak_reserved_plus_non_torch_bytes=peak,peak_allocated_bytes=allocated,peak_memory_fraction=fraction,
            projected_optimizer_updates=updates,eval_examples=eval_examples,full_eval_outer_steps=16,
            projected_training_hours=training_hours,projected_evaluation_hours=evaluation_hours,
            projected_compute_hours=training_hours+evaluation_hours,
            planning_hours_with_25_percent_margin=1.25*(training_hours+evaluation_hours),
            exclusions='Projection excludes compilation, repeated restarts, data loading outside timed batches, checkpoint I/O, CPU voting/aggregation and queue time. 25% is a planning allowance, not a guarantee.',
            config=raw,sources=source_hashes(),
            diagnostics_present=sorted(diagnostics))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value,indent=2),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()

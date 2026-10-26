"""Resumable native scheduled evaluation using the unchanged upstream ARC voter.

Metric accumulation follows vendor pretrain.evaluate. Cursor and local voter caches
are checkpointed together by the driver; replay skips batches without recounting.
"""
from pathlib import Path
import torch
import torch.distributed as dist


def evaluate(config, model, loader, metadata, evaluators, rank, world, cpu_group,
             progress, save, should_stop, device='cuda', checkpoint_every=128):
    if config.eval_save_outputs:
        raise ValueError('Reproduction evaluator does not support raw eval_save_outputs.')
    set_ids={name:i for i,name in enumerate(metadata.sets)}
    keys=set()
    for ev in evaluators:keys.update(ev.required_outputs)
    if not progress.get('started'):
        for ev in evaluators:ev.begin_eval()
        progress.update(started=True,cursor=0,metric_keys=[],metric_values=None)
    with torch.inference_mode():
        for index,(set_name,batch,_) in enumerate(loader):
            if index<progress['cursor']:continue
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.device(device):carry=model.initial_carry(batch)
            while True:
                carry,loss,metrics,preds,done=model(carry=carry,batch=batch,return_keys=keys)
                if done:break
            for ev in evaluators:ev.update_batch(batch,preds)
            if progress['metric_values'] is None:
                progress['metric_keys']=sorted(metrics)
                progress['metric_values']=torch.zeros((len(set_ids),len(metrics)),device=device)
            progress['metric_values'][set_ids[set_name]]+=torch.stack([metrics[k] for k in progress['metric_keys']])
            progress['cursor']=index+1
            del carry,loss,metrics,preds,batch
            stopping=should_stop()
            if stopping or progress['cursor']%checkpoint_every==0:save()
            if rank==0:print(f"Evaluation batch {progress['cursor']} committed to memory",flush=True)
            if stopping:return False,None
        # Commit complete local accumulators before the collective/final voting.
        # Reduce a clone: a restart must never reduce already globally summed data.
        save()
        values=progress['metric_values']
        reduced=None
        if values is not None:
            values=values.clone()
            if world>1:dist.reduce(values,dst=0)
            if rank==0:
                reduced={}
                for name,index in set_ids.items():
                    row=dict(zip(progress['metric_keys'],values[index].cpu().tolist()))
                    count=row.pop('count')
                    reduced[name]={k:v/count if count else 0.0 for k,v in row.items()}
        for ev in evaluators:
            path=Path(config.checkpoint_path)/f'evaluator_{type(ev).__name__}_step_{config.step}'
            path.mkdir(parents=True,exist_ok=True)
            metrics=ev.result(str(path),rank=rank,world_size=world,group=cpu_group)
            if rank==0 and metrics is not None:
                if reduced is None:reduced={}
                reduced.update(metrics)
        return True,reduced

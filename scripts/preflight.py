"""Four-rank single-node CUDA/NCCL/Gloo check; reject unintended GPU hardware."""
import datetime
import os
import re
import socket
import torch
import torch.distributed as dist


def main():
    if int(os.environ['WORLD_SIZE']) != 4 or int(os.environ['LOCAL_WORLD_SIZE']) != 4:
        raise RuntimeError('Expected four workers on one node.')
    if torch.cuda.device_count() != 4: raise RuntimeError('Each worker must see all four allocated GPUs.')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    name = torch.cuda.get_device_name()
    expected = os.environ.get('TRM_EXPECT_GPU', 'GH200').upper()
    if expected not in ('GH200', 'H200') or not re.search(r'\b'+expected+r'\b', name.upper()):
        raise RuntimeError(f'Expected {expected}, allocated {name}. Check Slurm resources and configs/cluster.env.sh.')
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    group = dist.new_group(backend='gloo', timeout=datetime.timedelta(minutes=5))
    value = torch.tensor(float(dist.get_rank()+1), device='cuda'); dist.all_reduce(value)
    if value.item() != 10: raise RuntimeError('Four-GPU NCCL all-reduce failed.')
    hosts = [None]*4; dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1: raise RuntimeError('All GPUs must be on the same node.')
    x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    x.square().mean().backward()
    if not torch.isfinite(x.grad).all().item(): raise RuntimeError('BF16 backward failed.')
    if dist.get_rank() == 0: print('PREFLIGHT PASSED: 1 node / 4 '+expected+' GPUs; '+torch.__version__, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()

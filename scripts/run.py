"""Host-side launcher: one baseline, one immutable configuration, no new project."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=160200)
    parser.add_argument('--max-updates', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()
    if args.seconds <= 600 or args.gpus != 4: parser.error('Need >600 seconds and exactly 4 GPUs.')
    config = args.config
    raw = json.loads(config.read_text()); out = Path(raw['checkpoint_path']); out.mkdir(parents=True, exist_ok=True)
    with (out/'launcher.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('This baseline is already running.')
        if (out/'COMPLETE.json').exists(): print('Baseline already completed.'); return 0
        if (out/'STOP').exists(): print('Baseline paused by STOP file.'); return 4
        begun = time.monotonic()
        subprocess.run([sys.executable, 'scripts/verify_vendor.py'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, 'scripts/repro_data.py', 'audit', '--dataset', 'sudoku'], cwd=ROOT, check=True)
        remaining = args.seconds-int(time.monotonic()-begun)-300
        if remaining <= 300: raise RuntimeError('Data audit used the available time; training was not launched.')
        def command(module, arguments):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            return ['bash', str(ROOT/'scripts/container.sh'),
                '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=4',
                '--master_addr=127.0.0.1', f'--master_port={port}', '--max_restarts=0',
                '-m', module, *arguments]
        subprocess.run(command('scripts.preflight', []), cwd=ROOT, check=True, timeout=min(900,remaining))
        remaining = args.seconds-int(time.monotonic()-begun)-300
        if remaining <= 300: raise RuntimeError('Preflight used the available time.')
        cmd = command('scripts.train', ['--config', str(config), '--seconds', str(remaining),
                      '--max-updates', str(args.max_updates), '--checkpoint-every', '1000'])
        pointer = out/'resume.json'; before = json.loads(pointer.read_text()) if pointer.exists() else None
        if before is not None:
            if before['world_size'] != 4: raise RuntimeError('Only matching four-GPU checkpoints can resume.')
            cmd.append('--resume')
        print('Launching native attention baseline:', raw['run_name'], flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode: raise SystemExit(result.returncode)
        if (out/'COMPLETE.json').exists(): return 0
        if not pointer.exists():
            raise RuntimeError('Worker exited without a committed checkpoint.')
        if (out/'STOP').exists(): return 4
        def progress(value):
            if value is None: return (0, 0, 0, 0)
            return (value['iteration'], int(value['phase'] == 'evaluation'), value['step'], value['evaluation_batches'])
        if progress(json.loads(pointer.read_text())) <= progress(before):
            raise RuntimeError('Worker exited without completing or committing progress.')
        if args.max_updates: return 4
        return 3  # Clean checkpoint; next allocation may continue.


if __name__ == '__main__': raise SystemExit(main())

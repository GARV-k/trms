"""Unchanged native optimizer/driver with explicit compilation and diagnostics."""
import argparse
import json
import os
from pathlib import Path
from scripts.experiment import make_config, source_hashes, ROOT


def main():
    p = argparse.ArgumentParser(add_help=False); p.add_argument('--config', type=Path, required=True)
    args, _ = p.parse_known_args()
    raw = json.loads(args.config.read_text()); mode = raw['study']['compile_mode']
    settings=raw['study']
    started = raw['run_name'].split('-sudoku-', 1)[0]
    expected=make_config(settings['paper_table_k'],settings['T'],settings['arm'],raw['global_batch_size'],raw['epochs'],settings['width'],mode=mode,started=started)
    if raw != expected: raise RuntimeError('Use an argparse-generated configuration unchanged.')
    if int(os.environ.get('WORLD_SIZE', '0')) != 4 or int(os.environ.get('LOCAL_WORLD_SIZE', '0')) != 4:
        raise RuntimeError('Expected exactly one node with four GPU workers.')
    from scripts import repro_train as driver
    from scripts.single4_compile import ManagedCompile
    original = driver.upstream.create_model
    def create_model(*args, **kwargs):
        previous = os.environ.get('DISABLE_COMPILE')
        os.environ['DISABLE_COMPILE'] = '1'
        try: model, optimizers, lrs = original(*args, **kwargs)
        finally:
            if previous is None: os.environ.pop('DISABLE_COMPILE', None)
            else: os.environ['DISABLE_COMPILE'] = previous
        # Same parameter objects and native optimizers; only execution is wrapped.
        return ManagedCompile(model, mode), optimizers, lrs
    driver.source_hashes = source_hashes
    driver.upstream.create_model = create_model
    print('Sudoku attention / width', settings['width'], '/ aug800; compilation:', mode, '; full checkpoints and three diagnostic plots enabled.', flush=True)
    try: driver.main()
    finally: driver.upstream.create_model = original


if __name__ == '__main__': main()

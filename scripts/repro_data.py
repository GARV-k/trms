"""Audit prepared data, or fetch/build the native Sudoku recipe offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import shutil

ROOT = Path(__file__).absolute().parents[1]
RAW = ROOT/'data/raw/sudoku-extreme'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def download_sudoku():
    RAW.mkdir(parents=True, exist_ok=True)
    receipt = RAW/'SOURCE.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        for name, value in record['sha256'].items():
            if digest(RAW/name) != value: raise RuntimeError('Raw data checksum mismatch: '+name)
        print('Raw Sudoku download already verified.'); return
    with urllib.request.urlopen('https://huggingface.co/api/datasets/sapientinc/sudoku-extreme') as r:
        revision = json.load(r)['sha']
    hashes = {}
    for name in ('train.csv', 'test.csv'):
        url = f'https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/{revision}/{name}'
        target = RAW/name; tmp = target.with_suffix('.download')
        with urllib.request.urlopen(url) as source, tmp.open('wb') as dest:
            while True:
                block = source.read(8*1024*1024)
                if not block: break
                dest.write(block)
        os.replace(tmp, target); hashes[name] = digest(target)
    receipt.write_text(json.dumps(dict(repo='sapientinc/sudoku-extreme', revision=revision, sha256=hashes), indent=2)+'\n')
    print('Pinned raw Sudoku dataset:', revision)


def build_sudoku():
    import numpy as np
    vendor = ROOT/'vendor/TinyRecursiveModels'
    sys.path[:0] = [str(vendor/'dataset'), str(vendor)]
    import build_sudoku_dataset as native
    out = ROOT/'data/paper/sudoku-extreme-1k-aug-800'
    if out.exists(): raise RuntimeError('Refusing to overwrite existing Sudoku data: '+str(out))
    record = json.loads((RAW/'SOURCE.json').read_text())
    for name, value in record['sha256'].items():
        if digest(RAW/name) != value: raise RuntimeError('Raw data checksum mismatch: '+name)
    # Preserve the native transformation code. Only replace its file download
    # with verified local files and fix its otherwise unspecified NumPy seed.
    native.hf_hub_download = lambda repo, name, **kwargs: str(RAW/name)
    np.random.seed(42)
    final = out
    out = final.with_name(final.name+'.building')
    if out.exists(): raise RuntimeError('Incomplete prior build; inspect and remove only this staging folder: '+str(out))
    config = native.DataProcessConfig(output_dir=str(out), subsample_size=1000, num_aug=800)
    native.convert_subset('train', config); native.convert_subset('test', config)
    metadata = {split: json.loads((out/split/'dataset.json').read_text()) for split in ('train', 'test')}
    hashes = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'DATA_READY.json').write_text(json.dumps(dict(dataset='sudoku', profile='paper',
        config=config.model_dump(), preprocessing_seed=42, raw_source=record, metadata=metadata, sha256=hashes), indent=2)+'\n')
    os.replace(out, final)
    print('Built native Sudoku 1K + 800 augmentations:', final)


def audit(dataset):
    folder = 'arc2concept-aug-800' if dataset == 'arc2' else 'sudoku-extreme-1k-aug-800'
    path = ROOT/'data/paper'/folder
    receipt = json.loads((path/'DATA_READY.json').read_text())
    if receipt['dataset'] != dataset or receipt['profile'] != 'paper' or receipt['config']['num_aug'] != 800:
        raise RuntimeError('Wrong data recipe.')
    if dataset == 'arc2':
        c = receipt['config']
        if c['seed'] != 42 or c['subsets'] != ['training2', 'evaluation2', 'concept'] or c['test_set_name'] != 'evaluation2':
            raise RuntimeError('ARC2 must use the native training2/evaluation2/concept split.')
    elif receipt['config']['subsample_size'] != 1000:
        raise RuntimeError('Sudoku must use 1000 original training examples.')
    if not receipt.get('sha256'): raise RuntimeError('Missing data checksums.')
    for name, value in receipt['sha256'].items():
        if digest(path/name) != value: raise RuntimeError('Prepared data checksum mismatch: '+name)
    print('Verified all prepared data checksums:', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['download-sudoku', 'build-sudoku', 'audit'])
    p.add_argument('--dataset', choices=['arc2', 'sudoku'])
    args = p.parse_args()
    if args.action == 'download-sudoku': download_sudoku()
    elif args.action == 'build-sudoku': build_sudoku()
    elif args.dataset: audit(args.dataset)
    else: p.error('audit requires --dataset')

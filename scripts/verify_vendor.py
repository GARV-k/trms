#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
root=Path(__file__).absolute().parents[1]
lock=json.loads((root/'repos.lock.json').read_text())
bad=[]
for name,digest in lock['files'].items():
    p=root/'vendor/TinyRecursiveModels'/name
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: bad.append(name)
if bad: raise SystemExit('Vendor differs from pinned source: '+', '.join(bad))
print('Verified official repository:',lock['commit'])

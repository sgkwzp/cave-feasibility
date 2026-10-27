import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
OEV = ROOT.parent / 'oev-feasibility'
TASKS = ['quesadilla', 'pinwheels']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    q.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

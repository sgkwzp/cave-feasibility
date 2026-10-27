"""Verify the published source/evidence snapshot without loading models or data."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'publication/files.json').read_text(encoding='utf-8'))
    version = json.loads((ROOT / 'VERSION.json').read_text(encoding='utf-8'))
    errors = []
    source_entries = []
    python_count = 0
    for entry in manifest['files']:
        relative = entry['path']
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'MISSING {relative}')
            continue
        if digest(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            errors.append(f'MISMATCH {relative}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
            python_count += 1
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
        if relative.startswith('source/') and path.suffix == '.py':
            source_entries.append(f"{entry['sha256']}  {relative}")
    source_sha = hashlib.sha256(('\n'.join(sorted(source_entries)) + '\n').encode()).hexdigest()
    if source_sha != version['source_set_sha256']:
        errors.append('SOURCE SET MISMATCH')
    primary = json.loads((ROOT / 'deployment-mechanism/summary.json').read_text(encoding='utf-8-sig'))
    aggregate = primary['aggregates']['all_five']
    checks = {
        'predicted_roi_increment': aggregate['P']['contrasts']['primary']['frame']['auc'],
        'predicted_vs_raw_aem': aggregate['P']['contrasts']['vs_raw_aem']['frame']['auc'],
        'gt_tail_vs_pre': aggregate['M']['contrasts']['primary']['macro']['auc'],
    }
    for key, value in checks.items():
        if version['evidence'][key] != value:
            errors.append(f'EVIDENCE MISMATCH {key}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({
        'status': 'verified', 'version': version['version'],
        'files': len(manifest['files']), 'python_files': python_count,
        'source_set_sha256': source_sha,
        'scope': 'published file hashes, Python syntax, JSON validity, evidence identity; no model execution',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

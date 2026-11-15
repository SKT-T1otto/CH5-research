"""Independent source/config preparation audit. Never import an algorithm or train."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'provenance/pure_marl_revision3_manifest.json'
EVOLUTION = 'provenance/pure_marl_revision3_evolution.json'
PREVIOUS = ('provenance/pure_marl_manifest.json', 'provenance/pure_marl_evolution.json',
            'provenance/pure_marl_revision2_manifest.json', 'provenance/pure_marl_revision2_evolution.json')
PARENT_SOURCES = 'provenance/pure_marl_parent_sources_v1.json'
PARENT = 'provenance/reach_avoid_monitor_manifest.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(root, name):
    return json.loads((root / name).read_text(encoding='utf-8'))


def inventory(root):
    parent = read(root, PARENT)
    names = set(parent['files'])
    for folder in ('core', 'chapter3_bser', 'configs', 'checks', 'tools'):
        names.update(p.relative_to(root).as_posix() for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json'))
    names.update(p.relative_to(root).as_posix() for p in (root / 'configs/pure_marl/upstream').rglob('*') if p.is_file())
    names.update((PARENT, PARENT_SOURCES, EVOLUTION, *PREVIOUS, 'provenance/pure_marl_parent_v1.json',
                  'docs/PURE_MARL_BASELINES_20261106.md', 'configs/pure_marl/requirements-harl.txt',
                  'configs/pure_marl/requirements-benchmarl.txt'))
    return {name: sha(root / name) for name in sorted(names)}


def protected_parent(root, files):
    parent = read(root, PARENT)
    receipt = read(root, 'provenance/pure_marl_parent_v1.json')
    originals = read(root, PARENT_SOURCES)['files']
    if (sha(root / PARENT) != receipt['parent_manifest_sha256']
            or parent['sha256'] != receipt['parent_review_sha256']
            or digest({k: v for k, v in parent.items() if k != 'sha256'}) != parent['sha256']):
        raise ValueError('parent source receipt changed')
    mutations = {
        'chapter3_bser/learning/provenance.py': ("""if (root/'provenance/pure_marl_manifest.json').is_file():
    from chapter3_bser.baselines.source import verify
    return verify(root,source_files(root),canonical_digest)""", 'provenance_ast_sha256'),
        'tools/verify_source.py': ("""if (root/'provenance/pure_marl_manifest.json').is_file():
    from tools.verify_pure_marl import verify as verify_pure_marl
    return verify_pure_marl(root)""", 'independent_verifier_ast_sha256'),
    }
    for name, original in parent['files'].items():
        if name not in mutations:
            if files.get(name) != original:
                raise ValueError('protected parent file changed: ' + name)
            continue
        code, key = mutations[name]
        statement = ast.dump(ast.parse(code).body[0], include_attributes=False)
        tree = ast.parse((root / name).read_text(encoding='utf-8'))
        removed = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Try)):
                removed += sum(ast.dump(n, include_attributes=False) == statement for n in node.body)
                node.body = [n for n in node.body if ast.dump(n, include_attributes=False) != statement]
        original = base64.b64decode(originals[name]['base64'], validate=True)
        if hashlib.sha256(original).hexdigest() != parent['files'][name]:
            raise ValueError('parent snapshot does not match immutable source: ' + name)
        if removed != 1 or ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(original.decode('utf-8')), include_attributes=False):
            raise ValueError('unreviewed parent verifier change: ' + name)
    exact = {PARENT, PARENT_SOURCES, EVOLUTION, *PREVIOUS, 'provenance/pure_marl_parent_v1.json', 'checks/pure_marl_contracts.py',
             'tools/fetch_pure_marl_upstream.py', 'tools/verify_pure_marl.py', 'docs/PURE_MARL_BASELINES_20261106.md'}
    extra = {n for n in files.keys() - parent['files'].keys()
             if not n.startswith(('chapter3_bser/baselines/', 'configs/pure_marl/')) and n not in exact}
    if extra:
        raise ValueError('unexpected additions: ' + repr(sorted(extra)))
    for name in files:
        if not name.startswith(('core/', 'chapter3_bser/')) or not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((root / name).read_text(encoding='utf-8'))):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            if any(module.split('.')[0] in ('tools', 'checks', 'tests') for module in modules):
                raise ValueError('runtime imports verification code: ' + name)


def record(root):
    """Write a new source evolution once; this is not a training-readiness receipt."""
    if (root / MANIFEST).exists() or (root / EVOLUTION).exists():
        raise FileExistsError('source evidence already exists; create a separately reviewed revision')
    # Build inventory with a temporary in-memory evolution placeholder.
    parent = read(root, PARENT)
    names = set(parent['files'])
    for folder in ('core', 'chapter3_bser', 'configs', 'checks', 'tools'):
        names.update(p.relative_to(root).as_posix() for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json'))
    names.update(p.relative_to(root).as_posix() for p in (root / 'configs/pure_marl/upstream').rglob('*') if p.is_file())
    names.update((PARENT, PARENT_SOURCES, *PREVIOUS, 'provenance/pure_marl_parent_v1.json', 'docs/PURE_MARL_BASELINES_20261106.md',
                  'configs/pure_marl/requirements-harl.txt', 'configs/pure_marl/requirements-benchmarl.txt'))
    files = {n: sha(root / n) for n in names}
    protected_parent(root, files)
    changes = []
    for name in sorted(parent['files'].keys() | files.keys()):
        before, after = parent['files'].get(name), files.get(name)
        if before != after:
            reason = ('Exact additive source-profile dispatch; parent implementation AST preserved'
                      if before else 'New pure-MARL design/config/interface/verification input; no parent algorithm edits')
            changes.append(dict(path=name, before_sha256=before, after_sha256=after, reason=reason))
    evolution = dict(schema='auv.pure_marl.evolution.v1', parent_review_sha256=parent['sha256'], changes=changes)
    with (root / EVOLUTION).open('x', encoding='utf-8') as stream:
        json.dump(evolution, stream, indent=2)
    data = dict(schema='auv.active_source.pure_marl_preparation.v1', files=inventory(root),
                parent_review_sha256=parent['sha256'], baseline_control_algorithm_added=True,
                parent_algorithms_unchanged=True, execution_enabled=False, native_backend_acceptance_passed=False,
                performance_passed=False, historical_acceptance_transferred=False,
                previous_preparation_records={name: sha(root / name) for name in PREVIOUS})
    data['sha256'] = digest(data)
    with (root / MANIFEST).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
    return verify(root)


def verify(root=ROOT):
    root = Path(root).resolve()
    data, parent = read(root, MANIFEST), read(root, PARENT)
    files = inventory(root)
    if (data.get('schema') != 'auv.active_source.pure_marl_preparation.v1'
            or data.get('sha256') != digest({k: v for k, v in data.items() if k != 'sha256'})
            or data.get('files') != files or data.get('execution_enabled') is not False):
        raise ValueError('pure-MARL source manifest/inventory mismatch')
    protected_parent(root, files)
    evolution = read(root, EVOLUTION)
    indexed = {r['path']: r for r in evolution['changes']}
    changed = {n for n in parent['files'].keys() | files.keys() if parent['files'].get(n) != files.get(n)} - {EVOLUTION}
    if len(indexed) != len(evolution['changes']) or set(indexed) != changed or evolution['parent_review_sha256'] != parent['sha256']:
        raise ValueError('incomplete source evolution')
    for name, item in indexed.items():
        if item['before_sha256'] != parent['files'].get(name) or item['after_sha256'] != files.get(name) or not item['reason']:
            raise ValueError('invalid source evolution hash: ' + name)
    lock = read(root, 'configs/pure_marl/upstream/lock.json')
    for backend in lock['sources'].values():
        for item in backend['files'].values():
            if sha(root / 'configs/pure_marl/upstream' / item['local']) != item['sha256']:
                raise ValueError('upstream configuration bytes changed')
    return dict(passed=True, scope='source_and_preparation_only', files=len(files),
                review_sha256=data['sha256'], parent_algorithms_unchanged=True, parent_evidence_immutable=True,
                training_started=False, native_backend_acceptance_passed=False, performance_passed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='create immutable first source record only')
    args = parser.parse_args()
    print(json.dumps(record(ROOT) if args.record else verify(), indent=2))

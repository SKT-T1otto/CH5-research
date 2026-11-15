"""Source identity for the additive, non-executable baseline preparation release."""
import ast
import base64
import hashlib
import json
from pathlib import Path

MANIFEST = 'provenance/pure_marl_revision3_manifest.json'
EVOLUTION = 'provenance/pure_marl_revision3_evolution.json'
PREVIOUS = ('provenance/pure_marl_manifest.json', 'provenance/pure_marl_evolution.json',
            'provenance/pure_marl_revision2_manifest.json', 'provenance/pure_marl_revision2_evolution.json')
PARENT_SOURCES = 'provenance/pure_marl_parent_sources_v1.json'
PARENT = 'provenance/reach_avoid_monitor_manifest.json'
PARENT_RECEIPT = 'provenance/pure_marl_parent_v1.json'
DISPATCH = """if (root/'provenance/pure_marl_manifest.json').is_file():
    from chapter3_bser.baselines.source import verify
    return verify(root,source_files(root),canonical_digest)"""
TOOL_DISPATCH = """if (root/'provenance/pure_marl_manifest.json').is_file():
    from tools.verify_pure_marl import verify as verify_pure_marl
    return verify_pure_marl(root)"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root, actual):
    root, files = Path(root), dict(actual)
    for name in (PARENT, PARENT_RECEIPT, PARENT_SOURCES, EVOLUTION, *PREVIOUS, 'docs/PURE_MARL_BASELINES_20261106.md',
                 'configs/pure_marl/requirements-harl.txt', 'configs/pure_marl/requirements-benchmarl.txt'):
        files[name] = sha(root / name)
    for path in (root / 'configs/pure_marl/upstream').rglob('*'):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha(path)
    return files


def assert_parent_unchanged(root, files, digest):
    root = Path(root)
    receipt = json.loads((root / PARENT_RECEIPT).read_text(encoding='utf-8'))
    parent = json.loads((root / PARENT).read_text(encoding='utf-8'))
    originals = json.loads((root / PARENT_SOURCES).read_text(encoding='utf-8'))['files']
    if (sha(root / PARENT) != receipt['parent_manifest_sha256']
            or parent['sha256'] != receipt['parent_review_sha256']
            or parent['sha256'] != digest({k: v for k, v in parent.items() if k != 'sha256'})):
        raise ValueError('immutable parent manifest changed')
    metadata = {'chapter3_bser/learning/provenance.py': (DISPATCH, 'provenance_ast_sha256'),
                'tools/verify_source.py': (TOOL_DISPATCH, 'independent_verifier_ast_sha256')}
    for filename, (dispatch, key) in metadata.items():
        target = ast.dump(ast.parse(dispatch).body[0], include_attributes=False)
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        removed = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Try)):
                removed.extend(n for n in node.body if ast.dump(n, include_attributes=False) == target)
                node.body = [n for n in node.body if ast.dump(n, include_attributes=False) != target]
        original = base64.b64decode(originals[filename]['base64'], validate=True)
        if hashlib.sha256(original).hexdigest() != parent['files'][filename]:
            raise ValueError('parent source snapshot bytes changed: ' + filename)
        if len(removed) != 1 or ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(original.decode('utf-8')), include_attributes=False):
            raise ValueError('parent source changed beyond exact additive dispatch: ' + filename)
    for name, expected in parent['files'].items():
        if name not in metadata and sha(root / name) != expected:
            raise ValueError('protected parent file changed: ' + name)
    allowed = ('chapter3_bser/baselines/', 'configs/pure_marl/')
    exact = {PARENT, PARENT_RECEIPT, PARENT_SOURCES, EVOLUTION, *PREVIOUS, 'checks/pure_marl_contracts.py',
             'tools/fetch_pure_marl_upstream.py', 'tools/verify_pure_marl.py',
             'docs/PURE_MARL_BASELINES_20261106.md'}
    unexpected = {p for p in files.keys() - parent['files'].keys() if not p.startswith(allowed) and p not in exact}
    if unexpected:
        raise ValueError('unreviewed baseline extension: ' + repr(sorted(unexpected)))


def verify(root, actual, canonical_digest):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    data, parent = read(MANIFEST), read(PARENT)
    files = inventory(root, actual)
    if (data.get('schema') != 'auv.active_source.pure_marl_preparation.v1'
            or data.get('sha256') != canonical_digest({k: v for k, v in data.items() if k != 'sha256'})
            or data.get('files') != files or data.get('execution_enabled') is not False):
        raise ValueError('baseline preparation source inventory/identity mismatch')
    assert_parent_unchanged(root, files, canonical_digest)
    evolution = read(EVOLUTION)
    indexed = {r['path']: r for r in evolution['changes']}
    required = {n for n in parent['files'].keys() | files.keys()
                if parent['files'].get(n) != files.get(n)} - {EVOLUTION}
    if (len(indexed) != len(evolution['changes']) or set(indexed) != required
            or evolution['parent_review_sha256'] != parent['sha256']):
        raise ValueError('baseline source evolution coverage mismatch')
    for name, row in indexed.items():
        if (row.get('before_sha256') != parent['files'].get(name)
                or row.get('after_sha256') != files.get(name) or not row.get('reason')):
            raise ValueError('baseline evolution hash mismatch: ' + name)
    return dict(review_sha256=data['sha256'], profile='active_pure_marl_preparation_v1',
                inventory_sha256=canonical_digest(files), parent_commit=parent.get('parent_commit',
                    'e613ecb05bd6ef6d38f306f85fd7079e66857774'), distribution=data['schema'])

"""Fetch pinned public configuration files only; never install or train."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'harl': ('PKU-MARL/HARL', 'b1af98b0dbab72a2eee9d160751cd09aedbb8ce2', [
        'harl/configs/algos_cfgs/maddpg.yaml', 'harl/configs/algos_cfgs/matd3.yaml',
        'harl/configs/algos_cfgs/mappo.yaml', 'harl/configs/algos_cfgs/happo.yaml',
        'setup.py',
    ]),
    'benchmarl': ('facebookresearch/BenchMARL', '65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1', [
        'benchmarl/conf/algorithm/masac.yaml',
        'benchmarl/conf/experiment/base_experiment.yaml',
        'benchmarl/conf/model/layers/mlp.yaml', 'LICENSE',
    ]),
}


def fetch(output, execute=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'configs/pure_marl'):
        raise ValueError('configuration snapshots must stay in configs/pure_marl')
    if not execute:
        return dict(executed=False, sources=SOURCES, output=str(output))
    if output.exists():
        raise FileExistsError(output)
    payloads, sources = {}, {}
    for backend, (repo, commit, paths) in SOURCES.items():
        entries = {}
        for remote in paths:
            url = f'https://raw.githubusercontent.com/{repo}/{commit}/{remote}'
            with urlopen(url, timeout=30) as response:
                data = response.read()
            name = f'{backend}/{remote}'
            payloads[name] = data
            entries[remote] = dict(local=name, url=url, sha256=hashlib.sha256(data).hexdigest())
        sources[backend] = dict(repository=f'https://github.com/{repo}', commit=commit, files=entries)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    with (output / 'lock.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(schema='auv.pure_marl.upstream.v1', sources=sources), stream, indent=2)
    return dict(executed=True, files=len(payloads), output=str(output), training_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'configs/pure_marl/upstream'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(fetch(args.output, args.execute), indent=2))

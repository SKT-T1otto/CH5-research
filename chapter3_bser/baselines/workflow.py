"""Prepare pure-MARL experiment files. This release never starts training."""
import argparse
import copy
import json
from pathlib import Path
from chapter3_bser.baselines.experiment import DEFAULT_SUITE, load_suite, load_splits, native_config, plan, read, sha, write
from chapter3_bser.baselines.evaluation import summarize
from chapter3_bser.learning.config import scenario_fingerprint
from core.scenarios.ch3_generator_impl import build_scenario_manifests


def prepare(suite_path, output, paths=None):
    from chapter3_bser.learning.provenance import verify_reviewed_changes
    source = verify_reviewed_changes()
    suite, reference, lock = load_suite(suite_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifests = {}
    if paths:
        scenes = load_splits(paths, reference)
        manifests = {key: read(path) for key, path in paths.items()}
    else:
        # Generation only: no simulator construction, no model initialization.
        for split, options in suite['scenario_splits'].items():
            origin = 'train' if split == 'train' else 'validation'
            raw = build_scenario_manifests(count=options['count'], generator_seed=options['generator_seed'],
                split=origin, profiles=[reference['profile']])[reference['profile']]
            manifest = copy.deepcopy(raw)
            role = 'train' if split == 'train' else 'eval'
            for scene in manifest['scenarios']:
                scene.update(scenario_role=role, scenario_split=role, max_steps=reference['max_steps'])
                # Distinguish two validation-role manifests in metadata only.
                scene['scenario_id'] = f"pure_marl_{split}_{scene['scenario_id']}"
            manifest.update(suite_split=split, scenario_role=role, scenario_split=role,
                            max_steps=reference['max_steps'], source_identity=source)
            manifests[split] = manifest
        # Validate in memory without writing partial output.
        from chapter3_bser.learning.scenarios import validate_manifest
        scenes = {split: validate_manifest(m, reference, evaluation=split != 'train') for split, m in manifests.items()}
        for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
            for field in ('scenario_id', 'scenario_seed'):
                if {s[field] for s in scenes[left]} & {s[field] for s in scenes[right]}:
                    raise ValueError(f'{left}/{right} overlap by {field}')
            if {scenario_fingerprint(s) for s in scenes[left]} & {scenario_fingerprint(s) for s in scenes[right]}:
                raise ValueError('physical scenario overlap')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'suite.json', suite)
    write(output / 'reference_config.json', reference)
    write(output / 'upstream_lock.json', lock)
    for split, manifest in manifests.items():
        write(output / f'{split}_scenarios.json', manifest)
    jobs = plan(suite_path)['jobs']
    for job in jobs:
        destination = output / 'jobs' / job['job_id']
        destination.mkdir(parents=True, exist_ok=False)
        spec = dict(job, schema='auv.pure_marl.job.v1', source_identity=source, stage='design_only',
                    execution_enabled=False, checkpoint_restore=None,
                    native_configuration=native_config(job['algorithm'], job['seed'], suite, destination / 'native_output'),
                    evaluation=suite['evaluation'],
                    manifests={key: dict(path=str(output / f'{key}_scenarios.json'),
                                        sha256=sha(output / f'{key}_scenarios.json')) for key in scenes})
        write(destination / 'run_spec.json', spec)
    receipt = dict(schema='auv.pure_marl.preparation.v1', source_identity=source, jobs=len(jobs),
                   scenario_counts={key: len(rows) for key, rows in scenes.items()},
                   training_started=False, checkpoint_loaded=False, execution_enabled=False,
                   native_backend_acceptance_passed=False, performance_passed=False,
                   files={p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()})
    write(output / 'preparation.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'report'))
    parser.add_argument('--suite', default=str(DEFAULT_SUITE))
    parser.add_argument('--output')
    parser.add_argument('--train-manifest')
    parser.add_argument('--dev-manifest')
    parser.add_argument('--test-manifest')
    parser.add_argument('--records', nargs='+')
    parser.add_argument('--execute', action='store_true', help='write preparation/report files only')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = plan(args.suite)
    elif not args.execute:
        result = dict(command=args.command, executed=False, training_started=False, output=args.output)
    elif args.command == 'prepare':
        if not args.output:
            parser.error('--output required')
        paths = dict(train=args.train_manifest, dev=args.dev_manifest, test=args.test_manifest)
        if any(paths.values()) and not all(paths.values()):
            parser.error('provide train/dev/test manifests together')
        result = prepare(args.suite, args.output, paths if all(paths.values()) else None)
    else:
        if not args.records or not args.output:
            parser.error('--records and --output required')
        rows = [json.loads(line) for path in args.records for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        result = summarize(rows)
        write(args.output, result)
    display = {k: v for k, v in result.items() if k != 'files'}
    if 'files' in result:
        display['files_count'] = len(result['files'])
        display['output'] = args.output
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()

"""Geometry and provenance only. Deliberately has no --train or --execute."""
import argparse
import json

from .config import DEFAULT_CONFIG, load, local_path, read_json, write_new
from .scenarios import build_registry, validate_registry, verify_regeneration


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--build-scenes", action="store_true")
    parser.add_argument("--regenerate-sample", action="store_true")
    parser.add_argument("--register-source", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--report")
    args = parser.parse_args()
    config = load(args.config)
    path = local_path(config["scene_manifest"])
    if args.build_scenes:
        if path.exists():
            raise FileExistsError(path)
        write_new(path, build_registry(config, args.config, args.workers))
    registry = read_json(path)
    report = dict(registry=validate_registry(config, registry, args.config))
    if args.regenerate_sample:
        report["regeneration"] = verify_regeneration(config, registry, args.workers)
    if args.build_scenes and not args.register_source:
        report["source"] = "NOT_REGISTERED; run --register-source after code checks"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.register_source:
        from .provenance import register
        report["source"] = register(config)
    else:
        from .provenance import verify
        report["source"] = verify(config)
    report.update(passed=True, training_episodes=0, task_rollouts=0, optimizer_updates=0)
    if args.report:
        write_new(local_path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Archive read-only registration inputs; never import the Chapter 4 project."""
import argparse
from pathlib import Path
import shutil
import subprocess

from chapter3_bser.learning.config import scenario_fingerprint
from chapter5_comm.nominal_maddpg.config import ROOT, file_sha, write_new, read_json

DESTINATION = ROOT / "provenance/ch5_standard_maddpg_20261125"
CH4_FILES = ("configs/ch4/nominal_maddpg.json",
             "chapter4_interval_robust/nominal_maddpg/config.py",
             "chapter4_interval_robust/nominal_maddpg/README.md",
             "chapter4_interval_robust/nominal_maddpg/runtime.py",
             "chapter4_interval_robust/scenarios.py",
             "experiments/ch4_work1_recoverability/plant_adapter.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-plan", type=Path, required=True)
    parser.add_argument("--development-plan", type=Path, required=True)
    parser.add_argument("--ch4-root", type=Path, required=True)
    args = parser.parse_args()
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    sources = {"historical_30": args.historical_plan, "development_24": args.development_plan}
    plans = {key: read_json(path) for key, path in sources.items()}
    if len(plans["historical_30"]["scenes"]) != 30 or len(plans["development_24"]["scenes"]) != 24:
        raise ValueError("expected the historical 30 and preregistered development 24")
    for relative in CH4_FILES:
        if not (args.ch4_root / relative).is_file():
            raise FileNotFoundError(args.ch4_root / relative)
    DESTINATION.mkdir(parents=True)
    cohorts, receipts = {}, {}
    for name, path in sources.items():
        plan = plans[name]
        dest = DESTINATION / "reference_inputs" / (name + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        # Physical innovation seeds in the historical batch differ from scene IDs.
        seeds = (dict(plan["seeds"]["typed_mapping"]) if name == "historical_30"
                 else dict(enumerate(plan["physical_seeds"])))
        cohorts[name] = dict(original_path=str(path.resolve()),
            archived_source=dest.relative_to(ROOT).as_posix(), source_sha256=file_sha(path),
            entries=[dict(index=i, scenario_id=s["scenario_id"], scenario_seed=s["scenario_seed"],
                          physical_seed=seeds[i], physical_fingerprint=scenario_fingerprint(s))
                     for i, s in enumerate(plan["scenes"])])
    for relative in CH4_FILES:
        source = args.ch4_root / relative
        dest = DESTINATION / "reference_inputs/ch4" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        receipts[relative] = dict(original_path=str(source.resolve()), sha256=file_sha(source),
                                 archived_path=dest.relative_to(ROOT).as_posix())
    write_new(DESTINATION / "history_exclusion.json", dict(schema="auv.ch5.maddpg.exclusions.v1",
        purpose="previously inspected scenes; regression only, never new admission/test", cohorts=cohorts))
    write_new(DESTINATION / "ch4_reference.json", dict(date="2026-11-25", files=receipts,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.ch4_root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=args.ch4_root, text=True),
        imported_at_runtime=False, plant_migration_approved=False,
        migration_status="REVIEW_REQUIRED: independent relative-water dynamics are not the CH5 native plant"))
    print("Archived 54 exclusions and read-only Chapter 4 specifications", flush=True)


if __name__ == "__main__":
    main()

"""Byte-level source identity and a portable source capsule, including dirty files."""
import importlib.metadata
import platform
import subprocess
import sys
import zipfile

from .config import ROOT, digest, file_sha, local_path, read_json, resolve, write_new

DIRECTORY = ROOT / "provenance/ch5_standard_maddpg_20261125"
MANIFEST = DIRECTORY / "source_manifest.json"


def source_paths():
    # Conservative envelope includes optional inherited modules. Inclusion does
    # not claim that a module was active or that a policy was trained.
    files = set()
    for name in ("core", "chapter3_bser", "chapter5_comm"):
        files.update(p for p in (ROOT / name).rglob("*.py") if "__pycache__" not in p.parts)
    files.update((ROOT / "configs").rglob("*.json"))
    files.update(p for p in (DIRECTORY / "reference_inputs").rglob("*") if p.is_file())
    files.update(DIRECTORY / name for name in ("history_exclusion.json", "ch4_reference.json"))
    files.update(ROOT / name for name in ("project_paths.py", "requirements.txt", ".gitattributes",
        "AGENTS.md", "tools/register_ch5_maddpg_inputs.py", "checks/test_ch5_nominal_registration.py",
        "tools/audit_ch5_maddpg_registration.py", "tools/executor_trace.py", "tools/__init__.py",
        "provenance/ch3_parent_snapshot.json", "checks/__init__.py", "checks/test_ch5_history_rules.py",
        "docs/CH5_STANDARD_MADDPG_STEP1_20261125.md"))
    # Record the transitive local modules actually loaded during registration,
    # including any helper outside the conservative package directories.
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = local_candidate(name)
            if path is not None:
                files.add(path)
    return sorted(files)


def local_candidate(name):
    from pathlib import Path
    path = Path(name).resolve()
    return path if path.is_relative_to(ROOT) and path.suffix == ".py" and path.is_file() else None


def compare_ch4(config, resolved):
    reference = read_json(DIRECTORY / "ch4_reference.json")
    for row in reference["files"].values():
        if file_sha(local_path(row["archived_path"])) != row["sha256"]:
            raise ValueError("archived Chapter 4 reference differs")
    ch4 = read_json(local_path(reference["files"]["configs/ch4/nominal_maddpg.json"]["archived_path"]))
    common = {}
    for key, value in config["algorithm"].items():
        if key == "revision":
            continue
        if key not in ch4["algorithm"] or digest(value) != digest(ch4["algorithm"][key]):
            raise ValueError("Chapter 4/5 shared algorithm specification differs: " + key)
        common[key] = value
    if digest(config["learning"]) != digest(ch4["learning"]):
        raise ValueError("registered Chapter 4/5 first-round learning settings differ")
    task = dict(resolved["native_task"], **config["task"])
    common_task = {}
    for key in sorted(task.keys() & ch4["task"].keys()):
        if digest(task[key]) != digest(ch4["task"][key]):
            raise ValueError("shared task contract differs: " + key)
        common_task[key] = task[key]
    return dict(algorithm_common_fields=common, learning_common_fields=config["learning"],
        task_common_fields=common_task, ch4_algorithm_revision=ch4["algorithm"]["revision"],
        ch5_algorithm_revision=config["algorithm"]["revision"],
        environment_differences=[
            dict(field="horizontal_dynamics", ch4="relative-water linear drag; water speed amplitude 0.10 m/s",
                 ch5="native sequential acceleration update; additive flow forcing amplitude 0.18; different units/meaning"),
            dict(field="actuator_memory", ch4=ch4["plant"]["lag_memory_coefficient"], ch5=0.0),
            dict(field="map_and_navigation", ch4="inherited shared-map D2 runtime",
                 ch5="causal_d2_v1; four receiver-local maps; own_history_v1/native"),
            dict(field="handoff_and_communication", ch4="inherited fixed reliable handoff",
                 ch5="all message types use costed TDMA; actual delivery required"),
            dict(field="new_scene_seed_blocks", ch4=[v["start_seed"] for v in ch4["scene_groups"].values()],
                 ch5=[v["start_seed"] for v in config["scene_groups"].values()])],
        plant_migrated=False, plant_migration_status="REVIEW_REQUIRED",
        migration_reason="different equations, units and actuator memory; requires an independent physical-factor study",
        runtime_sibling_imports=False, ch4_source_receipt=file_sha(DIRECTORY / "ch4_reference.json"))


def register(config):
    if MANIFEST.exists() or (DIRECTORY / "source_snapshot.zip").exists():
        raise FileExistsError("source registration already exists; use a new revision for changed bytes")
    # Import only this project's runtime, without constructing or advancing it.
    from chapter5_comm.runtime import CH5Runtime
    resolved = resolve(config)
    comparison = compare_ch4(config, resolved)
    files = {p.relative_to(ROOT).as_posix(): dict(sha256=file_sha(p), bytes=p.stat().st_size)
             for p in source_paths()}
    observed = sorted({p.relative_to(ROOT).as_posix() for m in list(sys.modules.values())
                       if (name := getattr(m, "__file__", None)) and (p := local_candidate(name))})
    import torch
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "torch", "matplotlib")},
        device="cpu", configured_cpu_threads=1, actual_torch_threads=torch.get_num_threads(),
        mission_steps=0, optimizer_updates=0)
    write_new(DIRECTORY / "resolved.json", resolved)
    write_new(DIRECTORY / "ch4_ch5_comparison.json", comparison)
    archive = DIRECTORY / "source_snapshot.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            raw = local_path(relative).read_bytes()
            import hashlib
            if hashlib.sha256(raw).hexdigest() != files[relative]["sha256"]:
                raise ValueError("source changed while archiving: " + relative)
            entry = zipfile.ZipInfo(relative, date_time=(2026, 11, 25, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, raw)
    record = dict(schema="auv.ch5.maddpg.byte_source.v1", date="2026-11-25",
        git_head_before_publication=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status_before_publication=subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True),
        identity_basis="raw file bytes; Git HEAD alone does not identify this workspace",
        source_tree_sha256=digest(files), files=files, observed_registration_imports=observed,
        coverage="conservative core/chapter3/chapter5 Python envelope, all JSON configs, observed local imports, and registration evidence",
        coverage_limit="not a dynamic rollout trace; optional/unused inherited algorithms are included",
        snapshot=dict(path=archive.relative_to(ROOT).as_posix(), sha256=file_sha(archive)),
        resolved_sha256=file_sha(DIRECTORY / "resolved.json"),
        comparison_sha256=file_sha(DIRECTORY / "ch4_ch5_comparison.json"), environment=environment,
        weights_loaded=[], task_rollouts=0, optimizer_updates=0)
    write_new(MANIFEST, record)
    return verify(config)


def verify(config):
    import hashlib
    from chapter5_comm.runtime import CH5Runtime
    record = read_json(MANIFEST)
    if digest(record["files"]) != record["source_tree_sha256"]:
        raise ValueError("source manifest identity differs")
    for relative, expected in record["files"].items():
        path = local_path(relative)
        if not path.is_file() or file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("dependency byte mismatch: " + relative)
    # New files under runtime/config roots cannot silently extend the envelope.
    if not {p.relative_to(ROOT).as_posix() for p in source_paths()} <= set(record["files"]):
        raise ValueError("new source dependencies require new registration")
    archive = local_path(record["snapshot"]["path"])
    if file_sha(archive) != record["snapshot"]["sha256"]:
        raise ValueError("source capsule bytes differ")
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(record["files"]):
            raise ValueError("source capsule inventory differs")
        for relative, expected in record["files"].items():
            if hashlib.sha256(bundle.read(relative)).hexdigest() != expected["sha256"]:
                raise ValueError("source capsule member differs: " + relative)
    resolved = resolve(config)
    for filename, field, actual in (("resolved.json", "resolved_sha256", resolved),
            ("ch4_ch5_comparison.json", "comparison_sha256", compare_ch4(config, resolved))):
        path = DIRECTORY / filename
        if file_sha(path) != record[field] or digest(read_json(path)) != digest(actual):
            raise ValueError("expanded configuration or comparison differs: " + filename)
    return dict(passed=True, source_files=len(record["files"]),
                source_tree_sha256=record["source_tree_sha256"], source_capsule_verified=True,
                manifest_sha256=file_sha(MANIFEST), weights_loaded=[], physics_steps=0, optimizer_updates=0)

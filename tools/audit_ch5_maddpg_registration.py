"""Finish Step 1 admission without training or mission rollouts."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from chapter5_comm.nominal_maddpg.config import ROOT, DEFAULT_CONFIG, digest, file_sha, load, local_path, read_json, write_new
from chapter5_comm.nominal_maddpg.provenance import DIRECTORY, register
from chapter5_comm.nominal_maddpg.scenarios import validate_registry, verify_regeneration


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): dict(sha256=file_sha(p), bytes=p.stat().st_size)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="outputs/chapter5/audits/standard_maddpg_step1_20261125")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    folder = local_path(args.audit)
    if (folder / "validation.json").exists():
        raise FileExistsError("acceptance receipt already exists")
    before = read_json(folder / "before.json")
    c = load()
    registry = read_json(local_path(c["scene_manifest"]))
    scenes = validate_registry(c, registry, DEFAULT_CONFIG)
    protocol = read_json(folder / "history_rule_tests.json")
    if not protocol["passed"] or protocol["exit_code"] != 0 or protocol["tests_run"] != 2:
        raise ValueError("existing protocol tests did not pass")
    for relative, expected in protocol["input_hashes"].items():
        if file_sha(local_path(relative)) != expected:
            raise ValueError("protocol source changed after its tests: " + relative)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-B", "-m", "unittest",
                             "checks.test_ch5_nominal_registration", "-v"], cwd=ROOT,
                            capture_output=True, text=True)
    log = result.stdout + result.stderr
    (folder / "registration_tests.txt").write_text(log, encoding="utf-8")
    print(log, flush=True)
    if result.returncode:
        raise RuntimeError("registration tests failed; source is not sealed")
    match = re.search(r"Ran (\d+) tests", log)
    if not match or int(match.group(1)) != 14 or "skipped" in log.lower():
        raise RuntimeError("all 14 registration tests must execute")
    tests = dict(passed=True, tests_run=int(match.group(1)), exit_code=result.returncode,
                 wall_seconds=time.perf_counter()-started, log_sha256=file_sha(folder / "registration_tests.txt"))
    regeneration = verify_regeneration(c, registry, args.workers)
    print("Independent regeneration passed", flush=True)
    # Existing project files must retain their opening bytes, including the
    # three pre-existing dependencies that will be versioned in this commit.
    changed = [relative for relative, row in before["project_before"].items()
               if not (ROOT / relative).is_file() or file_sha(ROOT / relative) != row["sha256"]]
    if changed:
        raise ValueError("pre-existing files changed during registration: " + repr(changed))
    protected = {}
    for name, expected in before["protected"].items():
        actual = snapshot(ROOT.parents[2] / name)
        if actual != expected:
            raise ValueError("protected directory bytes changed: " + name)
        protected[name] = dict(files=len(actual), bytes=sum(v["bytes"] for v in actual.values()),
                               before_sha256=digest(expected), after_sha256=digest(actual), unchanged=True)
    evidence = read_json(local_path(c["exclusion_manifest"]))
    original_inputs = {}
    for name, cohort in evidence["cohorts"].items():
        actual = file_sha(Path(cohort["original_path"]))
        if actual != cohort["source_sha256"]:
            raise ValueError("original history plan changed: " + name)
        original_inputs[name] = dict(sha256=actual, unchanged=True)
    source = register(c)
    report = dict(schema="auv.ch5.maddpg.step1.acceptance.v1", date="2026-11-25", passed=True,
        step1_registration_passed=True, config_parsed=True, scenes=scenes,
        registry_tests=tests, existing_protocol_tests=protocol, regeneration=regeneration,
        source=source, preexisting_project_files_checked=len(before["project_before"]),
        preexisting_project_byte_changes=changed, protected_directories=protected,
        original_history_inputs=original_inputs,
        ch4_plant_migrated=False, ch4_plant_migration_status="REVIEW_REQUIRED",
        existing_dependencies_versioned=["chapter5_comm/runtime.py", "chapter5_comm/history_rules.py",
                                        "checks/test_ch5_history_rules.py"],
        task_rollouts=0, mission_physics_steps=0, optimizer_updates=0, weights_loaded=[],
        learner_implemented=False, performance_admitted=False,
        next_allowed_stage="network construction only; separate validation required")
    write_new(folder / "validation.json", report)
    publish = ROOT / "verification/ch5_standard_maddpg_step1_20261125"
    publish.mkdir(parents=True, exist_ok=False)
    write_new(publish / "validation.json", report)
    text = f"""# 第五章标准 MADDPG 第 1 步验收

2026-11-25。登记验收通过。尚未实现或验收学习更新，尚未评价任务成功率。

- 标准单 Critic 配置完整解析；实际展开的物理、奖励、残差比例、期限、信道和地图路径已记录。
- 三个训练种子为 2729、3733、4751；场景共 1156 场，分别为训练 1000、验证 32、准入 100、链路故障开发 24。正式通信测试尚未登记。
- 全部物理指纹重算；跨组重复为 0，与历史 30 场及开发 24 场重复为 0。12 场按事前固定的各组首/中/末索引独立重建一致。
- 14 项登记测试及 2 项已有历史帧规则测试通过。后者仅检查初始化后的实际序列化及收发账本，没有推进任务物理步。
- {source['source_files']} 项源码/配置/登记证据逐项 SHA-256 绑定；压缩包内实际字节已核验。Git HEAD 与未提交状态另列，不能替代字节身份。
- 四五章共有算法与首轮学习规范逐字段一致；原生流场与相对水速模型、推进记忆、局部地图和交接不同。第四章动力学未迁入，迁移状态为 REVIEW_REQUIRED。
- {len(before['project_before'])} 个本轮开始时已有项目文件字节不变。CH3、CH4、CH5 共 {sum(v['files'] for v in protected.values())} 个受保护文件前后字节一致。
- 任务回合、任务物理推进、训练更新均为 0；没有加载权重。生成器的目标轨迹/真几何检查属于原有数据集可行性筛选，不是策略结果。

字节源树摘要为 `{source['source_tree_sha256']}`。

机器校验见同目录 `validation.json`；完整展开和来源包见 `provenance/ch5_standard_maddpg_20261125/`，详细规范见 `docs/CH5_STANDARD_MADDPG_STEP1_20261125.md`。这里只放行下一步网络构造，不放行训练或通信性能比较。
"""
    for dest in (folder / "REPORT.md", publish / "REPORT.md"):
        with dest.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    created = {name: dict(sha256=file_sha(ROOT / name), bytes=(ROOT / name).stat().st_size)
               for name in files if name and name not in before["project_before"] and (ROOT / name).is_file()}
    for path in folder.iterdir():
        if path.is_file():
            created[path.relative_to(ROOT).as_posix()] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
    receipt = dict(created_files=created,
        self_excluded=[(folder / "created_files.json").relative_to(ROOT).as_posix(),
                       (publish / "created_files.json").relative_to(ROOT).as_posix()],
        modified_preexisting_files=[], existing_dependencies_versioned=report["existing_dependencies_versioned"])
    write_new(folder / "created_files.json", receipt)
    write_new(publish / "created_files.json", receipt)
    print(json.dumps(dict(passed=True, report=str(folder / "REPORT.md"), source=source), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

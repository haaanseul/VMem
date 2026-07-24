from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_revisit_experiment.py"
GROUPS = ("yaw", "long")
MODES = ("surfel", "recent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run visually meaningful local revisit cycles. These are long-form "
            "diagnostics, not the official RealEstate10K benchmark."
        )
    )
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--scene", default="test_samples/oxford.jpg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/results/long_cycle")
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_specs(groups: list[str], modes: list[str]) -> list[dict]:
    specs: list[dict] = []
    if "yaw" in groups:
        for mode in modes:
            specs.append(
                {
                    "run_id": f"yaw_cycle_{mode}",
                    "group": "yaw",
                    "context_mode": mode,
                    "trajectory": "yaw_cycle",
                    "movement_steps": 1,
                    "yaw_step": 10,
                    "interp_frames": 7,
                    "cycle_yaw_steps": 9,
                    "cycle_turn_degrees": 90,
                    "expected_frames": 127,
                }
            )
    if "long" in groups:
        for mode in modes:
            specs.append(
                {
                    "run_id": f"long_reverse_cycle_{mode}",
                    "group": "long",
                    "context_mode": mode,
                    "trajectory": "long_reverse_cycle",
                    "movement_steps": 6,
                    "yaw_step": 5,
                    "interp_frames": 4,
                    "cycle_yaw_steps": 9,
                    "cycle_turn_degrees": 90,
                    "expected_frames": 433,
                }
            )
    return specs


def command_for(
    spec: dict,
    *,
    scene: str,
    seed: int,
    run_dir: Path,
    inference_steps: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--scene",
        scene,
        "--seed",
        str(seed),
        "--context-mode",
        spec["context_mode"],
        "--trajectory",
        spec["trajectory"],
        "--memory-intervention",
        "correct",
        "--movement-steps",
        str(spec["movement_steps"]),
        "--yaw-step",
        str(spec["yaw_step"]),
        "--interp-frames",
        str(spec["interp_frames"]),
        "--cycle-yaw-steps",
        str(spec["cycle_yaw_steps"]),
        "--cycle-turn-degrees",
        str(spec["cycle_turn_degrees"]),
        "--output-dir",
        str(run_dir.relative_to(ROOT)),
    ]
    if inference_steps is not None:
        command.extend(("--inference-steps", str(inference_steps)))
    return command


def load_summary(run_dir: Path) -> dict | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_suite_summary(output_dir: Path, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite_summary.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    if ROOT.resolve() not in output_dir.parents:
        raise ValueError("--output-dir must be inside the VMem repository")
    scene_path = (ROOT / args.scene).resolve()
    if ROOT.resolve() not in scene_path.parents or not scene_path.exists():
        raise ValueError("--scene must be an existing file inside the VMem repository")

    specs = build_specs(list(args.groups), list(args.modes))
    print(
        f"[long-suite] planned_runs={len(specs)} output={output_dir}",
        flush=True,
    )
    suite_summary_path = output_dir / "suite_summary.json"
    if suite_summary_path.exists() and not args.force:
        existing_records = load_json(suite_summary_path)
    else:
        existing_records = []
    records_by_id = {
        record["run_id"]: record
        for record in existing_records
        if record.get("run_id")
    }
    failures = 0

    for index, spec in enumerate(specs, start=1):
        run_dir = output_dir / spec["run_id"]
        command = command_for(
            spec,
            scene=str(scene_path.relative_to(ROOT)),
            seed=args.seed,
            run_dir=run_dir,
            inference_steps=args.inference_steps,
        )
        if args.dry_run:
            print(
                f"[{index}/{len(specs)}] expected_frames={spec['expected_frames']} "
                + " ".join(command),
                flush=True,
            )
            continue

        existing = load_summary(run_dir)
        if existing is not None and not args.force:
            status = (
                "existing"
                if existing.get("num_frames") == spec["expected_frames"]
                else "frame_count_mismatch"
            )
            records_by_id[spec["run_id"]] = {
                **spec,
                "status": status,
                "actual_frames": existing.get("num_frames"),
                "runtime_seconds": existing.get("runtime_seconds"),
                "fallback_count": existing.get("retrieval", {}).get(
                    "fallback_count"
                ),
            }
            print(
                f"[{index}/{len(specs)}] {status} {spec['run_id']}",
                flush=True,
            )
            if status != "existing":
                failures += 1
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(specs)}] start {spec['run_id']} "
            f"expected_frames={spec['expected_frames']}",
            flush=True,
        )
        started = time.perf_counter()
        with (run_dir / "console.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - started
        summary = load_summary(run_dir)
        actual_frames = summary.get("num_frames") if summary else None
        status = (
            "complete"
            if result.returncode == 0
            and actual_frames == spec["expected_frames"]
            else "failed"
        )
        if status == "failed":
            failures += 1
        records_by_id[spec["run_id"]] = {
            **spec,
            "status": status,
            "actual_frames": actual_frames,
            "suite_wall_seconds": round(elapsed, 3),
            "runtime_seconds": summary.get("runtime_seconds") if summary else None,
            "fallback_count": (
                summary.get("retrieval", {}).get("fallback_count")
                if summary
                else None
            ),
        }
        save_suite_summary(output_dir, list(records_by_id.values()))
        print(
            f"[{index}/{len(specs)}] {status} {spec['run_id']} "
            f"frames={actual_frames} seconds={elapsed:.1f}",
            flush=True,
        )

    if not args.dry_run:
        save_suite_summary(output_dir, list(records_by_id.values()))
    if failures:
        raise SystemExit(f"{failures} long revisit run(s) failed")


if __name__ == "__main__":
    main()

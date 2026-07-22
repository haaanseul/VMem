from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_revisit_experiment.py"
GROUPS = (
    "dose",
    "components",
    "novel",
    "partial",
    "context",
    "rotation",
    "gap",
    "contamination",
    "repro",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded VMem revisit follow-up matrix sequentially."
    )
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--output-dir", default="experiments/results/overnight")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=("test_samples/living_room.jpg", "test_samples/open_door.jpg"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def add_common(spec: dict) -> dict:
    return {
        "scene": "test_samples/living_room.jpg",
        "seed": 42,
        "context_mode": "surfel",
        "trajectory": "exact_revisit",
        "memory_intervention": "correct",
        "intervention_components": "latent_clip",
        "movement_steps": 1,
        "interp_frames": 1,
        **spec,
    }


def build_specs(groups: list[str], scenes: list[str], seeds: list[int]) -> list[dict]:
    specs = []
    selected = set(groups)

    if "dose" in selected:
        for count in (0, 1, 2, 4):
            specs.append(
                add_common(
                    {
                        "group": "dose",
                        "run_id": f"dose_slots{count}",
                        "memory_intervention": "correct_plus_wrong",
                        "wrong_slot_count": count,
                    }
                )
            )

    if "components" in selected:
        for component in ("latent", "clip", "latent_clip", "pose_intrinsics"):
            specs.append(
                add_common(
                    {
                        "group": "components",
                        "run_id": f"component_{component}_slots1",
                        "memory_intervention": "correct_plus_wrong",
                        "intervention_components": component,
                        "wrong_slot_count": 1,
                    }
                )
            )

    if "novel" in selected:
        for yaw in (10, 20, 30):
            for intervention in ("correct", "wrong", "correct_plus_wrong"):
                specs.append(
                    add_common(
                        {
                            "group": "novel",
                            "run_id": f"novel_yaw{yaw}_{intervention}",
                            "trajectory": "novel_angle_revisit",
                            "revisit_yaw_offset": yaw,
                            "memory_intervention": intervention,
                        }
                    )
                )

    if "partial" in selected:
        for intervention in ("correct", "wrong", "correct_plus_wrong"):
            specs.append(
                add_common(
                    {
                        "group": "partial",
                        "run_id": f"partial_yaw45_{intervention}",
                        "trajectory": "partial_overlap",
                        "revisit_yaw_offset": 45,
                        "memory_intervention": intervention,
                    }
                )
            )

    if "context" in selected:
        for mode in ("surfel", "recent", "initial_only"):
            specs.append(
                add_common(
                    {
                        "group": "context",
                        "run_id": f"context_{mode}_movement4",
                        "context_mode": mode,
                        "movement_steps": 4,
                    }
                )
            )
        specs.append(
            add_common(
                {
                    "group": "context",
                    "run_id": "context_none_movement4",
                    "context_mode": "surfel",
                    "movement_steps": 4,
                    "memory_intervention": "none",
                }
            )
        )
        for mode in ("recent", "initial_only"):
            specs.append(
                add_common(
                    {
                        "group": "context",
                        "run_id": f"novel_context_{mode}_yaw20",
                        "context_mode": mode,
                        "trajectory": "novel_angle_revisit",
                        "revisit_yaw_offset": 20,
                    }
                )
            )
        for mode in ("surfel", "recent", "initial_only"):
            specs.append(
                add_common(
                    {
                        "group": "context",
                        "run_id": f"novel_context_{mode}_movement4_yaw20",
                        "context_mode": mode,
                        "trajectory": "novel_angle_revisit",
                        "movement_steps": 4,
                        "revisit_yaw_offset": 20,
                    }
                )
            )

    if "rotation" in selected:
        for schedule in ("90x1", "45x2", "30x3", "15x6"):
            specs.append(
                add_common(
                    {
                        "group": "rotation",
                        "run_id": f"rotation_{schedule}",
                        "trajectory": "rotation_accumulation",
                        "rotation_schedule": schedule,
                    }
                )
            )

    if "gap" in selected:
        for gap in ("short", "medium", "long"):
            specs.append(
                add_common(
                    {
                        "group": "gap",
                        "run_id": f"gap_{gap}",
                        "trajectory": "revisit_gap",
                        "revisit_gap": gap,
                    }
                )
            )

    if "contamination" in selected:
        for intervention in ("correct", "wrong", "correct_plus_wrong"):
            specs.append(
                add_common(
                    {
                        "group": "contamination",
                        "run_id": f"contamination_{intervention}",
                        "trajectory": "contamination_followup",
                        "memory_intervention": intervention,
                    }
                )
            )

    if "repro" in selected:
        for scene in scenes:
            scene_name = Path(scene).stem
            for seed in seeds:
                for intervention in ("correct", "wrong", "correct_plus_wrong"):
                    specs.append(
                        add_common(
                            {
                                "group": "repro",
                                "run_id": f"repro_{scene_name}_s{seed}_{intervention}",
                                "scene": scene,
                                "seed": seed,
                                "memory_intervention": intervention,
                            }
                        )
                    )
    return specs


def spec_to_command(spec: dict, output_dir: Path, inference_steps: int | None):
    command = [
        sys.executable,
        str(RUNNER),
        "--scene",
        spec["scene"],
        "--seed",
        str(spec["seed"]),
        "--context-mode",
        spec["context_mode"],
        "--trajectory",
        spec["trajectory"],
        "--memory-intervention",
        spec["memory_intervention"],
        "--intervention-components",
        spec["intervention_components"],
        "--movement-steps",
        str(spec["movement_steps"]),
        "--interp-frames",
        str(spec["interp_frames"]),
        "--output-dir",
        str(output_dir.relative_to(ROOT)),
    ]
    optional = {
        "wrong_slot_count": "--wrong-slot-count",
        "revisit_yaw_offset": "--revisit-yaw-offset",
        "rotation_schedule": "--rotation-schedule",
        "revisit_gap": "--revisit-gap",
    }
    for key, flag in optional.items():
        if key in spec:
            command.extend((flag, str(spec[key])))
    if inference_steps is not None:
        command.extend(("--inference-steps", str(inference_steps)))
    return command


def aggregate_record(spec: dict, run_dir: Path, status: str, seconds: float) -> dict:
    record = {
        "group": spec["group"],
        "run_id": spec["run_id"],
        "status": status,
        "runtime_wall_seconds": round(seconds, 3),
        "scene": spec["scene"],
        "seed": spec["seed"],
        "context_mode": spec["context_mode"],
        "trajectory": spec["trajectory"],
        "memory_intervention": spec["memory_intervention"],
        "intervention_components": spec["intervention_components"],
        "wrong_slot_count": spec.get("wrong_slot_count"),
        "revisit_yaw_offset": spec.get("revisit_yaw_offset"),
        "rotation_schedule": spec.get("rotation_schedule"),
        "revisit_gap": spec.get("revisit_gap"),
    }
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record.update(
            {
                "revisit_psnr": summary.get("revisit_consistency", {}).get("psnr"),
                "revisit_mae": summary.get("revisit_consistency", {}).get("mae"),
                "fallback_count": summary.get("retrieval", {}).get("fallback_count"),
                "validity_flag_count": len(
                    summary.get("rollout_validity", {}).get("flagged_frames", [])
                ),
                "generation_calls": summary.get("generation_calls"),
                "gpu_peak_memory_mb": summary.get("gpu_peak_memory_mb"),
            }
        )
    return record


def save_aggregate(output_dir: Path, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix_summary.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fieldnames = sorted({key for record in records for key in record})
    with (output_dir / "matrix_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    if ROOT.resolve() not in output_dir.parents:
        raise ValueError("--output-dir must be inside the VMem repository")
    specs = build_specs(list(args.groups), list(args.scenes), list(args.seeds))
    print(f"[matrix] planned runs={len(specs)} output={output_dir}", flush=True)
    aggregate_path = output_dir / "matrix_summary.json"
    if aggregate_path.exists() and not args.force:
        existing_records = json.loads(aggregate_path.read_text(encoding="utf-8"))
    else:
        existing_records = []
    records_by_id = {record["run_id"]: record for record in existing_records}
    failures = 0

    for index, spec in enumerate(specs, start=1):
        run_dir = output_dir / spec["run_id"]
        command = spec_to_command(spec, run_dir, args.inference_steps)
        summary_path = run_dir / "summary.json"
        if args.dry_run:
            print(f"[{index}/{len(specs)}] {' '.join(command)}", flush=True)
            continue
        if summary_path.exists() and not args.force:
            print(f"[{index}/{len(specs)}] skip completed {spec['run_id']}", flush=True)
            records_by_id[spec["run_id"]] = aggregate_record(
                spec, run_dir, "existing", 0.0
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(specs)}] start {spec['run_id']}", flush=True)
        started = time.perf_counter()
        with (run_dir / "console.log").open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        seconds = time.perf_counter() - started
        status = "complete" if result.returncode == 0 and summary_path.exists() else "failed"
        if status == "failed":
            failures += 1
        records_by_id[spec["run_id"]] = aggregate_record(
            spec, run_dir, status, seconds
        )
        save_aggregate(output_dir, list(records_by_id.values()))
        print(
            f"[{index}/{len(specs)}] {status} {spec['run_id']} seconds={seconds:.1f}",
            flush=True,
        )

    if not args.dry_run:
        save_aggregate(output_dir, list(records_by_id.values()))
    if failures:
        raise SystemExit(f"{failures} matrix run(s) failed")


if __name__ == "__main__":
    main()

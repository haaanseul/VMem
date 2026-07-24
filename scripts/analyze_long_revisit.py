from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.revisit_experiment_utils import (  # noqa: E402
    image_metrics,
    make_contact_sheet,
    rotation_error_degrees,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match every revisit frame to the nearest outbound camera pose and "
            "measure paired cycle consistency."
        )
    )
    parser.add_argument(
        "--results-dir", default="experiments/results/long_cycle"
    )
    parser.add_argument("--max-contact-pairs", type=int, default=16)
    return parser


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frame_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "frames").glob("frame_*.png"))


def pose_match(
    return_pose: np.ndarray,
    outbound_poses: list[np.ndarray],
) -> tuple[int, float, float]:
    best: tuple[float, int, float, float] | None = None
    for index, pose in enumerate(outbound_poses):
        rotation = rotation_error_degrees(pose, return_pose)
        translation = float(np.linalg.norm(pose[:3, 3] - return_pose[:3, 3]))
        score = rotation + 100.0 * translation
        candidate = (score, index, rotation, translation)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No outbound poses available for cycle matching")
    return best[1], best[2], best[3]


def analyze_run(run_dir: Path, max_contact_pairs: int) -> dict:
    summary = load_json(run_dir / "summary.json")
    trajectory = load_json(run_dir / "trajectory.json")
    paths = frame_paths(run_dir)
    poses = [np.asarray(pose, dtype=np.float64) for pose in trajectory["actual_camera_poses"]]
    if len(paths) != len(poses):
        raise ValueError(
            f"{run_dir.name}: frame/pose count mismatch {len(paths)} != {len(poses)}"
        )

    revisit_records = [
        record
        for record in trajectory["command_records"]
        if record.get("phase") == "revisit"
    ]
    if not revisit_records:
        raise ValueError(f"{run_dir.name}: no revisit command records")
    revisit_start = min(int(record["frame_start"]) for record in revisit_records)
    outbound_poses = poses[:revisit_start]
    frames = [Image.open(path).convert("RGB") for path in paths]
    records: list[dict] = []

    for return_index in range(revisit_start, len(frames)):
        match_index, rotation_error, translation_error = pose_match(
            poses[return_index], outbound_poses
        )
        metrics = image_metrics(frames[match_index], frames[return_index])
        records.append(
            {
                "return_frame": return_index,
                "matched_outbound_frame": match_index,
                "pose_rotation_error_degrees": rotation_error,
                "pose_translation_error": translation_error,
                **metrics,
            }
        )

    psnr_values = np.asarray([record["psnr"] for record in records], dtype=np.float64)
    mae_values = np.asarray([record["mae"] for record in records], dtype=np.float64)
    result = {
        "run_id": run_dir.name,
        "context_mode": summary.get("context_mode"),
        "trajectory": summary.get("trajectory"),
        "num_frames": len(frames),
        "revisit_start_frame": revisit_start,
        "revisit_frame_count": len(records),
        "paired_cycle_consistency": {
            "mean_psnr": float(np.mean(psnr_values)),
            "median_psnr": float(np.median(psnr_values)),
            "min_psnr": float(np.min(psnr_values)),
            "mean_mae": float(np.mean(mae_values)),
            "max_mae": float(np.max(mae_values)),
            "final_psnr": records[-1]["psnr"],
            "final_mae": records[-1]["mae"],
        },
        "fallback_count": summary.get("retrieval", {}).get("fallback_count"),
        "automatic_validity_flag_count": len(
            summary.get("rollout_validity", {}).get("flagged_frames", [])
        ),
        "pairs": records,
    }
    save_json(run_dir / "cycle_pairs.json", result)

    with (run_dir / "cycle_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    pair_count = min(max_contact_pairs, len(records))
    sample_indices = np.linspace(0, len(records) - 1, pair_count, dtype=int)
    contact_images: list[Image.Image] = []
    contact_labels: list[str] = []
    for record_index in sample_indices:
        record = records[int(record_index)]
        outbound_index = int(record["matched_outbound_frame"])
        return_index = int(record["return_frame"])
        contact_images.extend((frames[outbound_index], frames[return_index]))
        contact_labels.extend(
            (
                f"outbound {outbound_index}",
                f"return {return_index} psnr={record['psnr']:.2f}",
            )
        )
    make_contact_sheet(
        contact_images,
        contact_labels,
        run_dir / "cycle_pair_contact_sheet.jpg",
        columns=2,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    results_dir = (ROOT / args.results_dir).resolve()
    if ROOT.resolve() not in results_dir.parents:
        raise ValueError("--results-dir must be inside the VMem repository")
    run_dirs = sorted(
        path.parent
        for path in results_dir.glob("*/summary.json")
        if (path.parent / "trajectory.json").exists()
    )
    analyses = []
    for run_dir in run_dirs:
        try:
            analysis = analyze_run(run_dir, args.max_contact_pairs)
        except ValueError as error:
            print(f"[long-analysis] skip {run_dir.name}: {error}", flush=True)
            continue
        analyses.append(analysis)
        paired = analysis["paired_cycle_consistency"]
        print(
            f"[long-analysis] {run_dir.name} "
            f"mean={paired['mean_psnr']:.2f}dB "
            f"final={paired['final_psnr']:.2f}dB",
            flush=True,
        )

    save_json(results_dir / "long_cycle_analysis.json", analyses)
    lines = ["# Long local cycle analysis", ""]
    lines.append(
        "These local command cycles are visually meaningful diagnostics, not the "
        "official RealEstate10K/Tanks-and-Temples benchmark."
    )
    lines.extend(
        [
            "",
            "| Run | Frames | Return frames | Mean paired PSNR | Minimum | Final | Fallback | Auto flags |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for analysis in analyses:
        paired = analysis["paired_cycle_consistency"]
        lines.append(
            f"| {analysis['run_id']} | {analysis['num_frames']} | "
            f"{analysis['revisit_frame_count']} | {paired['mean_psnr']:.2f} | "
            f"{paired['min_psnr']:.2f} | {paired['final_psnr']:.2f} | "
            f"{analysis['fallback_count']} | "
            f"{analysis['automatic_validity_flag_count']} |"
        )
    (results_dir / "long_cycle_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

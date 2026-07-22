from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.revisit_experiment_utils import image_metrics, save_json  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary(results: Path, run_id: str):
    return load_json(results / run_id / "summary.json")


def final_frame(results: Path, run_id: str) -> Image.Image:
    paths = sorted((results / run_id / "frames").glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frames for {run_id}")
    return Image.open(paths[-1]).convert("RGB")


def compact_metrics(metrics: dict) -> dict:
    return {
        "psnr": metrics.get("psnr"),
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
    }


def analyze(results: Path) -> dict:
    report = {}

    report["dose_response"] = {}
    for count in (0, 1, 2, 4):
        metrics = summary(results, f"dose_slots{count}")["revisit_consistency"]
        report["dose_response"][str(count)] = compact_metrics(metrics)

    report["component_ablation"] = {}
    for component in ("latent", "clip", "latent_clip", "pose_intrinsics"):
        metrics = summary(results, f"component_{component}_slots1")[
            "revisit_consistency"
        ]
        report["component_ablation"][component] = compact_metrics(metrics)

    report["novel_memory_sensitivity"] = {}
    for yaw in (10, 20, 30):
        reference = final_frame(results, f"novel_yaw{yaw}_correct")
        report["novel_memory_sensitivity"][str(yaw)] = {
            intervention: compact_metrics(
                image_metrics(
                    reference,
                    final_frame(results, f"novel_yaw{yaw}_{intervention}"),
                )
            )
            for intervention in ("wrong", "correct_plus_wrong")
        }

    partial_reference = final_frame(results, "partial_yaw45_correct")
    report["partial_overlap_sensitivity"] = {
        intervention: compact_metrics(
            image_metrics(
                partial_reference,
                final_frame(results, f"partial_yaw45_{intervention}"),
            )
        )
        for intervention in ("wrong", "correct_plus_wrong")
    }

    report["exact_context_baseline"] = {}
    context_runs = {
        "surfel": "context_surfel_movement4",
        "none_on_revisit": "context_none_movement4",
        "recent": "context_recent_movement4",
        "initial_only": "context_initial_only_movement4",
    }
    for label, run_id in context_runs.items():
        run_summary = summary(results, run_id)
        report["exact_context_baseline"][label] = {
            **compact_metrics(run_summary["revisit_consistency"]),
            "fallback_count": run_summary["retrieval"]["fallback_count"],
            "validity_flag_count": len(
                run_summary["rollout_validity"]["flagged_frames"]
            ),
        }

    long_novel_reference = final_frame(
        results, "novel_context_surfel_movement4_yaw20"
    )
    report["long_novel_context_sensitivity"] = {
        mode: compact_metrics(
            image_metrics(
                long_novel_reference,
                final_frame(results, f"novel_context_{mode}_movement4_yaw20"),
            )
        )
        for mode in ("recent", "initial_only")
    }

    rotation_reference = final_frame(results, "rotation_90x1")
    report["rotation_accumulation"] = {
        schedule: compact_metrics(
            image_metrics(
                rotation_reference,
                final_frame(results, f"rotation_{schedule}"),
            )
        )
        for schedule in ("45x2", "30x3", "15x6")
    }

    report["revisit_gap"] = {}
    for gap in ("short", "medium", "long"):
        run_summary = summary(results, f"gap_{gap}")
        report["revisit_gap"][gap] = {
            **compact_metrics(run_summary["revisit_consistency"]),
            "generation_calls": run_summary["generation_calls"],
            "fallback_count": run_summary["retrieval"]["fallback_count"],
        }

    report["contamination_persistence"] = {}
    for intervention in ("correct", "wrong", "correct_plus_wrong"):
        run_id = f"contamination_{intervention}"
        trajectory = load_json(results / run_id / "trajectory.json")
        run_summary = summary(results, run_id)
        report["contamination_persistence"][intervention] = {
            "immediate_revisit": compact_metrics(
                trajectory["command_records"][1]["last_frame_initial_metrics"]
            ),
            "post_intervention_followup": compact_metrics(
                run_summary["revisit_consistency"]
            ),
        }

    repro_values = {
        intervention: []
        for intervention in ("correct", "wrong", "correct_plus_wrong")
    }
    fallback_free_values = {
        intervention: []
        for intervention in ("correct", "wrong", "correct_plus_wrong")
    }
    fallback_free_scene_seeds = []
    per_scene = {}
    for scene in ("living_room", "open_door"):
        per_scene[scene] = {}
        for intervention in repro_values:
            values = []
            for seed in (42, 43, 44):
                value = summary(
                    results, f"repro_{scene}_s{seed}_{intervention}"
                )["revisit_consistency"]["psnr"]
                values.append(value)
                repro_values[intervention].append(value)
            per_scene[scene][intervention] = {
                "values": values,
                "mean": statistics.mean(values),
                "pstdev": statistics.pstdev(values),
            }
        scene_seed_summaries = {
            seed: {
                intervention: summary(
                    results, f"repro_{scene}_s{seed}_{intervention}"
                )
                for intervention in repro_values
            }
            for seed in (42, 43, 44)
        }
        for seed, summaries in scene_seed_summaries.items():
            if all(
                value["retrieval"]["fallback_count"] == 0
                for value in summaries.values()
            ):
                fallback_free_scene_seeds.append({"scene": scene, "seed": seed})
                for intervention, value in summaries.items():
                    fallback_free_values[intervention].append(
                        value["revisit_consistency"]["psnr"]
                    )
    report["reproducibility"] = {"per_scene": per_scene, "overall": {}}
    for intervention, values in repro_values.items():
        report["reproducibility"]["overall"][intervention] = {
            "values": values,
            "mean": statistics.mean(values),
            "pstdev": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    correct_mean = report["reproducibility"]["overall"]["correct"]["mean"]
    for intervention in ("wrong", "correct_plus_wrong"):
        report["reproducibility"]["overall"][intervention][
            "mean_drop_from_correct"
        ] = correct_mean - report["reproducibility"]["overall"][intervention][
            "mean"
        ]
    report["reproducibility"]["fallback_free"] = {
        "scene_seeds": fallback_free_scene_seeds,
        "conditions": {},
    }
    for intervention, values in fallback_free_values.items():
        report["reproducibility"]["fallback_free"]["conditions"][intervention] = {
            "values": values,
            "mean": statistics.mean(values),
            "pstdev": statistics.pstdev(values),
        }
    fallback_free_correct = report["reproducibility"]["fallback_free"][
        "conditions"
    ]["correct"]["mean"]
    for intervention in ("wrong", "correct_plus_wrong"):
        values = report["reproducibility"]["fallback_free"]["conditions"][
            intervention
        ]
        values["mean_drop_from_correct"] = fallback_free_correct - values["mean"]
    return report


def markdown(report: dict) -> str:
    lines = ["# Revisit matrix analysis", ""]
    lines.append("## Reproducibility")
    lines.append("")
    for name, values in report["reproducibility"]["overall"].items():
        suffix = ""
        if "mean_drop_from_correct" in values:
            suffix = f", drop={values['mean_drop_from_correct']:.2f} dB"
        lines.append(
            f"- {name}: mean={values['mean']:.2f} dB, "
            f"std={values['pstdev']:.2f} dB{suffix}"
        )
    fallback_free = report["reproducibility"]["fallback_free"]
    lines.append(
        f"- fallback-free subset: {len(fallback_free['scene_seeds'])} scene/seed pairs"
    )
    for name, values in fallback_free["conditions"].items():
        suffix = ""
        if "mean_drop_from_correct" in values:
            suffix = f", drop={values['mean_drop_from_correct']:.2f} dB"
        lines.append(
            f"  - {name}: mean={values['mean']:.2f} dB, "
            f"std={values['pstdev']:.2f} dB{suffix}"
        )
    lines.extend(["", "## Dose response", ""])
    for count, values in report["dose_response"].items():
        lines.append(f"- wrong slots {count}: {values['psnr']:.2f} dB")
    lines.extend(["", "## Component ablation", ""])
    for component, values in report["component_ablation"].items():
        lines.append(f"- {component}: {values['psnr']:.2f} dB")
    lines.extend(["", "## Exact context baseline", ""])
    for mode, values in report["exact_context_baseline"].items():
        lines.append(f"- {mode}: {values['psnr']:.2f} dB")
    lines.extend(["", "## Rotation accumulation", ""])
    for schedule, values in report["rotation_accumulation"].items():
        lines.append(f"- 90x1 vs {schedule}: {values['psnr']:.2f} dB")
    lines.extend(["", "## Contamination persistence", ""])
    for intervention, values in report["contamination_persistence"].items():
        lines.append(
            f"- {intervention}: immediate={values['immediate_revisit']['psnr']:.2f} dB, "
            f"followup={values['post_intervention_followup']['psnr']:.2f} dB"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results/overnight")
    args = parser.parse_args()
    results = (ROOT / args.results_dir).resolve()
    if ROOT.resolve() not in results.parents:
        raise ValueError("--results-dir must be inside the repository")
    report = analyze(results)
    save_json(results / "analysis.json", report)
    (results / "analysis.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")


if __name__ == "__main__":
    main()

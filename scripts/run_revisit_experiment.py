from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.revisit_experiment_utils import (  # noqa: E402
    GAP_REPETITIONS,
    ROTATION_SCHEDULES,
    TRAJECTORIES,
    build_trajectory,
    ensure_within_repo,
    frame_validity_metrics,
    image_metrics,
    make_contact_sheet,
    parse_command,
    rotation_error_degrees,
    save_json,
    save_jsonl,
    save_manual_manifest,
    save_summary_csv,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run instrumented VMem spatial-memory revisit experiments without Gradio."
    )
    parser.add_argument("--config", default="configs/inference/inference.yaml")
    parser.add_argument("--scene", default="test_samples/living_room.jpg")
    parser.add_argument(
        "--context-mode", choices=("surfel", "recent", "initial_only"), default="surfel"
    )
    parser.add_argument("--trajectory", choices=TRAJECTORIES, default="exact_revisit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--movement-step", type=float, default=0.1)
    parser.add_argument("--movement-steps", type=int, default=1)
    parser.add_argument("--yaw-step", type=float, default=15.0)
    parser.add_argument("--interp-frames", type=int, default=1)
    parser.add_argument("--revisit-gap", choices=tuple(GAP_REPETITIONS), default="short")
    parser.add_argument("--revisit-yaw-offset", type=float, default=10.0)
    parser.add_argument(
        "--rotation-schedule", choices=tuple(ROTATION_SCHEDULES), default="90x1"
    )
    parser.add_argument(
        "--memory-intervention",
        choices=("correct", "none", "wrong", "correct_plus_wrong"),
        default="correct",
    )
    parser.add_argument(
        "--intervention-phase",
        choices=("revisit", "all", "never"),
        default="revisit",
        help="Apply memory intervention only on revisit commands, on all commands, or never.",
    )
    parser.add_argument("--output-dir", default="experiments/results/dry_run")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Optional sampler-step override for smoke tests; omit for the checked-in config.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--list-trajectories", action="store_true")
    return parser


def resolve_repo_path(raw_path: str, *, must_exist: bool = False) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = ensure_within_repo(path, ROOT)
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def should_activate_intervention(phase: str, setting: str) -> bool:
    if setting == "all":
        return True
    if setting == "never":
        return False
    return phase in {"revisit", "revisit_offset"}


def run_navigation_command(navigator, command: str):
    name, amount = parse_command(command)
    if name == "yaw":
        amount = 10.0 if amount is None else amount
        return navigator.turn_left(amount) if amount >= 0 else navigator.turn_right(abs(amount))
    if name in {"forward", "f"}:
        return navigator.move_forward(int(1 if amount is None else amount))
    if name in {"back", "backward", "b"}:
        return navigator.move_backward(int(1 if amount is None else amount))
    raise ValueError(f"Unsupported experiment command: {command}")


def select_contact_frames(frames, maximum: int = 64):
    if len(frames) <= maximum:
        return list(enumerate(frames))
    indices = np.linspace(0, len(frames) - 1, maximum, dtype=int)
    return [(int(index), frames[int(index)]) for index in indices]


def main() -> None:
    args = build_parser().parse_args()
    if args.list_trajectories:
        print("\n".join(TRAJECTORIES))
        return
    if args.interp_frames < 1:
        raise ValueError("--interp-frames must be at least 1")

    scene_path = resolve_repo_path(args.scene, must_exist=True)
    config_path = resolve_repo_path(args.config, must_exist=True)
    output_dir = resolve_repo_path(args.output_dir)
    commands = build_trajectory(
        args.trajectory,
        movement_steps=args.movement_steps,
        yaw_step=args.yaw_step,
        revisit_gap=args.revisit_gap,
        revisit_yaw_offset=args.revisit_yaw_offset,
        rotation_schedule=args.rotation_schedule,
    )
    run_config = {
        "scene": str(scene_path.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "context_mode": args.context_mode,
        "trajectory": args.trajectory,
        "seed": args.seed,
        "movement_step": args.movement_step,
        "movement_steps": args.movement_steps,
        "yaw_step": args.yaw_step,
        "interp_frames": args.interp_frames,
        "revisit_gap": args.revisit_gap,
        "revisit_yaw_offset": args.revisit_yaw_offset,
        "rotation_schedule": args.rotation_schedule,
        "memory_intervention": args.memory_intervention,
        "intervention_phase": args.intervention_phase,
        "fps": args.fps,
        "inference_steps_override": args.inference_steps,
        "commands": commands,
        "plan_only": bool(args.plan_only),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", run_config)
    save_json(output_dir / "trajectory.json", {"planned_commands": commands})
    if args.plan_only:
        print(json.dumps(run_config, indent=2, ensure_ascii=False))
        return

    os.chdir(ROOT)
    seed_everything(args.seed)

    import torch
    from diffusers.utils import export_to_video
    from omegaconf import OmegaConf

    from modeling.pipeline import VMemPipeline
    from navigation import Navigator
    from utils import get_default_intrinsics, load_img_and_K, tensor_to_pil, transform_img_and_K

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VMem generation dry run")

    device = torch.device("cuda")
    config = OmegaConf.load(config_path)
    config.inference.visualize = False
    config.inference.visualize_pointcloud = False
    config.inference.visualize_surfel = False
    if args.inference_steps is not None:
        if args.inference_steps < 1:
            raise ValueError("--inference-steps must be at least 1")
        config.model.inference_num_steps = args.inference_steps
    run_config["resolved_model_config"] = OmegaConf.to_container(config, resolve=True)
    save_json(output_dir / "run_config.json", run_config)

    run_started = time.perf_counter()
    try:
        model = VMemPipeline(config, device)
        model.configure_revisit_experiment(
            context_mode=args.context_mode,
            memory_intervention=args.memory_intervention,
            intervention_active=args.intervention_phase == "all",
        )
        navigator = Navigator(
            model,
            step_size=args.movement_step,
            num_interpolation_frames=args.interp_frames,
        )

        image, _ = load_img_and_K(str(scene_path), None, K=None, device=device)
        image, _ = transform_img_and_K(
            image,
            (config.model.height, config.model.width),
            mode="crop",
            K=None,
        )
        initial_frame = tensor_to_pil(image.detach().cpu())
        initial_pose = np.eye(4, dtype=np.float32)
        initial_K = np.asarray(get_default_intrinsics()[0], dtype=np.float32)
        navigator.initialize(initial_frame, initial_pose, initial_K)

        torch.cuda.reset_peak_memory_stats(device)
        all_frames = [initial_frame]
        context_images = []
        context_labels = []
        command_records = []

        for command_plan in commands:
            active = should_activate_intervention(
                command_plan["phase"], args.intervention_phase
            )
            model.set_revisit_intervention_active(active)
            before_trace_count = len(model.debug_context_history)
            before_frame_count = len(all_frames)
            command_started = time.perf_counter()
            new_frames = run_navigation_command(navigator, command_plan["command"]) or []
            command_seconds = time.perf_counter() - command_started
            all_frames.extend(new_frames)

            new_entries = model.debug_context_history[before_trace_count:]
            for local_index, entry in enumerate(new_entries):
                trace_index = before_trace_count + local_index
                entry.update(
                    {
                        "command_index": command_plan["command_index"],
                        "command": command_plan["command"],
                        "phase": command_plan["phase"],
                        "command_seconds": command_seconds,
                        "generated_frame_start": before_frame_count,
                        "generated_frame_end": len(all_frames) - 1,
                    }
                )
                saved_context_paths = []
                for slot, (route_index, source_index) in enumerate(
                    zip(entry.get("route_indices", []), entry.get("content_source_indices", []))
                ):
                    context_image = model.pil_frames[int(source_index)]
                    relative_path = Path("contexts") / (
                        f"trace_{trace_index:04d}_slot_{slot:02d}_"
                        f"route_{int(route_index):04d}_source_{int(source_index):04d}.png"
                    )
                    context_path = output_dir / relative_path
                    context_path.parent.mkdir(parents=True, exist_ok=True)
                    context_image.save(context_path)
                    saved_context_paths.append(str(relative_path))
                    context_images.append(context_image.copy())
                    context_labels.append(
                        f"t{trace_index} s{slot} r{int(route_index)}→c{int(source_index)}"
                    )
                entry["context_image_paths"] = saved_context_paths

            command_records.append(
                {
                    **command_plan,
                    "intervention_active": active,
                    "generated_frames": len(new_frames),
                    "generation_calls": len(new_entries),
                    "runtime_seconds": command_seconds,
                    "frame_start": before_frame_count,
                    "frame_end": len(all_frames) - 1,
                }
            )
            print(
                "[experiment] "
                f"command={command_plan['command']} phase={command_plan['phase']} "
                f"frames={len(new_frames)} calls={len(new_entries)} "
                f"intervention={args.memory_intervention if active else 'correct'}",
                flush=True,
            )

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(all_frames):
            frame.save(frames_dir / f"frame_{index:04d}.png")

        export_to_video(all_frames, str(output_dir / "revisit.mp4"), fps=args.fps)
        actual_poses = [np.asarray(pose).tolist() for pose in model.c2ws]
        save_json(
            output_dir / "trajectory.json",
            {
                "planned_commands": commands,
                "command_records": command_records,
                "actual_camera_poses": actual_poses,
            },
        )
        save_json(output_dir / "memory_trace.json", model.debug_context_history)
        save_jsonl(output_dir / "memory_trace.jsonl", model.debug_context_history)

        contact_frames = select_contact_frames(all_frames)
        make_contact_sheet(
            [frame for _, frame in contact_frames],
            [f"frame {index}" for index, _ in contact_frames],
            output_dir / "frame_contact_sheet.jpg",
        )
        make_contact_sheet(
            context_images[:64],
            context_labels[:64],
            output_dir / "selected_memory_contact_sheet.jpg",
        )

        validity = []
        previous = None
        flagged_frames = []
        for index, frame in enumerate(all_frames):
            metrics = frame_validity_metrics(frame, previous)
            metrics["frame_index"] = index
            validity.append(metrics)
            if (
                not metrics["finite"]
                or metrics["black_candidate"]
                or metrics["saturation_candidate"]
                or metrics["abrupt_change_candidate"]
            ):
                flagged_frames.append(index)
            previous = frame

        final_pose = np.asarray(navigator.current_pose)
        runtime_seconds = time.perf_counter() - run_started
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        summary = {
            "scene": str(scene_path.relative_to(ROOT)),
            "seed": args.seed,
            "context_mode": args.context_mode,
            "trajectory": args.trajectory,
            "memory_intervention": args.memory_intervention,
            "intervention_phase": args.intervention_phase,
            "num_commands": len(commands),
            "generation_calls": len(model.debug_context_history),
            "num_frames": len(all_frames),
            "runtime_seconds": runtime_seconds,
            "gpu_peak_memory_mb": peak_memory_mb,
            "final_pose_error": {
                "rotation_degrees": rotation_error_degrees(initial_pose, final_pose),
                "translation": float(np.linalg.norm(final_pose[:3, 3] - initial_pose[:3, 3])),
            },
            "revisit_consistency": image_metrics(initial_frame, all_frames[-1]),
            "retrieval": {
                "fallback_count": sum(
                    1 for entry in model.debug_context_history if entry.get("fallback_used")
                ),
                "context_reasons": [
                    entry.get("reason") for entry in model.debug_context_history
                ],
            },
            "rollout_validity": {
                "flagged_frames": flagged_frames,
                "per_frame": validity,
                "automatic_labels_are_candidates_only": True,
            },
        }
        save_json(output_dir / "summary.json", summary)
        save_summary_csv(output_dir / "summary.csv", summary)
        save_manual_manifest(output_dir / "manual_labels.csv", len(all_frames))
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        save_json(
            output_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - run_started,
            },
        )
        raise


if __name__ == "__main__":
    main()

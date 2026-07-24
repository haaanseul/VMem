from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper demo's real Navigator/Pipeline path from either the "
            "current tree or an extracted official upstream source snapshot."
        )
    )
    parser.add_argument(
        "--implementation",
        choices=("current", "official"),
        required=True,
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Source tree to import. Defaults to the repository for current and "
            "experiments/upstream_reference for official."
        ),
    )
    parser.add_argument("--source-commit", default=None)
    parser.add_argument(
        "--compatibility-note",
        default=None,
        help="Non-behavioral source compatibility override applied to the snapshot.",
    )
    parser.add_argument(
        "--official-vae-repo",
        default="sd2-community/stable-diffusion-2-1",
        help=(
            "Runtime replacement for the retired official VAE repository ID. "
            "Used only with --implementation official."
        ),
    )
    parser.add_argument("--scene", default="test_samples/oxford.jpg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--veer-clicks",
        type=int,
        default=3,
        help="Number of official 20-degree Veer button clicks in each direction.",
    )
    parser.add_argument(
        "--effective-yaw",
        type=float,
        default=10.0,
        help=(
            "Effective Navigator yaw per click. The official app labels the "
            "button 20 degrees but passes y_angle//2, so the default is 10."
        ),
    )
    parser.add_argument("--interpolation-frames", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=13)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument(
        "--scene-reconstruction-mode",
        choices=("source_default", "recent_window", "full_history"),
        default="source_default",
        help=(
            "Override the current pipeline's CUT3R write window. The official "
            "source always uses full history."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def resolve_inside_repo(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path must stay inside the VMem repository: {path}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def image_metrics(reference: Image.Image, candidate: Image.Image) -> dict:
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    candidate_array = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    difference = reference_array - candidate_array
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def make_contact_sheet(
    frames: list[Image.Image],
    output_path: Path,
    maximum: int = 16,
) -> None:
    sample_count = min(maximum, len(frames))
    indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    width, height = frames[0].size
    label_height = 28
    columns = 4
    rows = int(np.ceil(sample_count / columns))
    sheet = Image.new(
        "RGB",
        (columns * width, rows * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for slot, frame_index in enumerate(indices):
        row, column = divmod(slot, columns)
        x = column * width
        y = row * (height + label_height)
        sheet.paste(frames[int(frame_index)].convert("RGB"), (x, y))
        draw.text((x + 8, y + height + 6), f"frame {int(frame_index)}", fill="black")
    sheet.save(output_path, quality=90)


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.veer_clicks < 1:
        raise ValueError("--veer-clicks must be at least 1")
    if args.interpolation_frames < 1:
        raise ValueError("--interpolation-frames must be at least 1")
    if args.effective_yaw <= 0:
        raise ValueError("--effective-yaw must be positive")

    default_source = (
        ROOT
        if args.implementation == "current"
        else ROOT / "experiments" / "upstream_reference"
    )
    source_root = resolve_inside_repo(
        args.source_root or str(default_source),
        must_exist=True,
    )
    scene_path = resolve_inside_repo(args.scene, must_exist=True)
    output_dir = resolve_inside_repo(args.output_dir)
    if source_root != ROOT and source_root in output_dir.parents:
        raise ValueError("--output-dir must not be inside the source snapshot")

    commands = [
        {
            "name": "turn_left",
            "degrees": args.effective_yaw,
            "button_label": "20° Veer",
        }
        for _ in range(args.veer_clicks)
    ]
    commands.extend(
        {
            "name": "turn_right",
            "degrees": args.effective_yaw,
            "button_label": "20° Veer",
        }
        for _ in range(args.veer_clicks)
    )
    plan = {
        "implementation": args.implementation,
        "source_root": str(source_root.relative_to(ROOT)),
        "source_commit": args.source_commit,
        "compatibility_note": args.compatibility_note,
        "official_vae_repository_override": (
            args.official_vae_repo
            if args.implementation == "official"
            else None
        ),
        "scene": str(scene_path.relative_to(ROOT)),
        "seed": args.seed,
        "veer_clicks_each_direction": args.veer_clicks,
        "official_button_label_degrees": 20,
        "effective_yaw_degrees_per_click": args.effective_yaw,
        "interpolation_frames": args.interpolation_frames,
        "step_size": args.step_size,
        "fps": args.fps,
        "inference_steps_override": args.inference_steps,
        "scene_reconstruction_mode": args.scene_reconstruction_mode,
        "commands": commands,
        "expected_real_frames": 1 + len(commands) * args.interpolation_frames,
        "visualization_disabled_for_io_only": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    os.chdir(source_root)
    sys.path.insert(0, str(source_root))

    import torch
    from diffusers.utils import export_to_video
    from omegaconf import OmegaConf

    if args.implementation == "official":
        from diffusers.models import AutoencoderKL

        original_from_pretrained = AutoencoderKL.from_pretrained

        def compatible_from_pretrained(repository, *positional, **keyword):
            if repository == "stabilityai/stable-diffusion-2-1-base":
                repository = args.official_vae_repo
            return original_from_pretrained(repository, *positional, **keyword)

        AutoencoderKL.from_pretrained = staticmethod(compatible_from_pretrained)

    from modeling.pipeline import VMemPipeline
    from navigation import Navigator
    from utils import (
        get_default_intrinsics,
        load_img_and_K,
        tensor_to_pil,
        transform_img_and_K,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper-demo reference run")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    config = OmegaConf.load(source_root / "configs/inference/inference.yaml")
    config.inference.visualize = False
    config.inference.visualize_pointcloud = False
    config.inference.visualize_surfel = False
    if args.inference_steps is not None:
        if args.inference_steps < 1:
            raise ValueError("--inference-steps must be at least 1")
        config.model.inference_num_steps = args.inference_steps
    if args.scene_reconstruction_mode != "source_default":
        config.inference.scene_reconstruction_mode = args.scene_reconstruction_mode
    plan["resolved_model_config"] = OmegaConf.to_container(config, resolve=True)
    save_json(output_dir / "run_config.json", plan)

    started = time.perf_counter()
    model = VMemPipeline(config, device)
    navigator = Navigator(
        model,
        step_size=args.step_size,
        num_interpolation_frames=args.interpolation_frames,
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

    command_records: list[dict] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast("cuda"):
        navigator.initialize(initial_frame, initial_pose, initial_K)
        for command_index, command in enumerate(commands):
            before_count = len(model.pil_frames)
            command_started = time.perf_counter()
            if command["name"] == "turn_left":
                returned_frames = navigator.turn_left(command["degrees"])
            else:
                returned_frames = navigator.turn_right(command["degrees"])
            command_records.append(
                {
                    "command_index": command_index,
                    **command,
                    "pipeline_frames_before": before_count,
                    "pipeline_frames_after": len(model.pil_frames),
                    "navigator_returned_frame_count": len(returned_frames or []),
                    "seconds": time.perf_counter() - command_started,
                }
            )

    frames = [frame.convert("RGB") for frame in model.pil_frames]
    expected_frames = int(plan["expected_real_frames"])
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} pipeline frames, got {len(frames)}"
        )

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{frame_index:04d}.png")
    export_to_video(frames, str(output_dir / "revisit.mp4"), fps=args.fps)
    make_contact_sheet(frames, output_dir / "frame_contact_sheet.jpg")

    poses = [np.asarray(pose, dtype=np.float64) for pose in model.c2ws]
    final_pose = poses[-1]
    final_rotation_trace = np.trace(final_pose[:3, :3])
    final_rotation_error = float(
        np.degrees(
            np.arccos(np.clip((final_rotation_trace - 1.0) / 2.0, -1.0, 1.0))
        )
    )
    final_translation_error = float(np.linalg.norm(final_pose[:3, 3]))
    summary = {
        **plan,
        "num_frames": len(frames),
        "num_commands": len(commands),
        "runtime_seconds": time.perf_counter() - started,
        "gpu_peak_memory_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "initial_to_final": image_metrics(frames[0], frames[-1]),
        "final_pose_error": {
            "rotation_degrees": final_rotation_error,
            "translation": final_translation_error,
        },
        "num_pipeline_latents": len(model.latents),
        "num_surfels": len(model.surfels),
        "command_records": command_records,
    }
    save_json(output_dir / "summary.json", summary)
    save_json(
        output_dir / "trajectory.json",
        {
            "camera_poses": [pose.tolist() for pose in poses],
            "command_records": command_records,
        },
    )
    with (output_dir / "command_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(command_records[0]))
        writer.writeheader()
        writer.writerows(command_records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

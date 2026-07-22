from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_command(command: str):
    name, _, value = command.partition(":")
    name = name.strip().lower()
    value = value.strip()
    amount = float(value) if value else None
    return name, amount


def rotation_error_degrees(reference_pose: np.ndarray, current_pose: np.ndarray) -> float:
    relative_rotation = reference_pose[:3, :3].T @ current_pose[:3, :3]
    trace_value = np.trace(relative_rotation)
    cos_value = np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def fov_degrees_from_intrinsics(K: np.ndarray) -> float:
    fx = float(K[0, 0])
    return float(np.degrees(2.0 * np.arctan(0.5 / fx)))


def image_metrics(reference, candidate):
    ref = np.asarray(reference).astype(np.float32) / 255.0
    cur = np.asarray(candidate).astype(np.float32) / 255.0
    mse = float(np.mean((ref - cur) ** 2))
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    mae = float(np.mean(np.abs(ref - cur)))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def save_dense_camera_poses(c2ws, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, pose in enumerate(c2ws):
        pose_array = np.asarray(pose)
        frames.append(
            {
                "file_path": f"images/frame_{idx:04d}.png",
                "transform_matrix": pose_array.tolist(),
            }
        )
    with open(output_path, "w") as f:
        json.dump({"frames": frames}, f, indent=2)


def make_preset_commands(preset: str, hfov_deg: float, occlude_margin: float):
    if preset == "yaw-return":
        return ["yaw:10"] * 9 + ["yaw:-10"] * 9
    if preset == "wide-yaw-return":
        yaw_deg = (hfov_deg / 2.0) + occlude_margin
        step = max(3.0, yaw_deg / 12.0)
        steps = int(np.ceil(yaw_deg / step))
        step = yaw_deg / steps
        return [f"yaw:{step:.3f}"] * steps + [f"yaw:{-step:.3f}"] * steps
    if preset == "forward-back":
        return ["forward:1"] * 8 + ["backward:1"] * 8
    if preset == "box-return":
        return (
            ["forward:1"] * 8
            + ["yaw:5"] * 18
            + ["forward:1"] * 8
            + ["yaw:5"] * 18
            + ["forward:1"] * 8
            + ["yaw:5"] * 18
            + ["forward:1"] * 8
            + ["yaw:5"] * 18
        )
    raise ValueError(f"Unknown preset: {preset}")


def run_command(navigator: Navigator, command: str):
    name, amount = parse_command(command)
    print(f"[revisit] command={command}", flush=True)

    if name == "yaw":
        if amount is None:
            amount = 10
        return navigator.turn_left(amount) if amount >= 0 else navigator.turn_right(abs(amount))
    if name in {"left", "l"}:
        return navigator.turn_left(amount if amount is not None else 10)
    if name in {"right", "r"}:
        return navigator.turn_right(amount if amount is not None else 10)
    if name in {"forward", "f"}:
        return navigator.move_forward(int(amount if amount is not None else 1))
    if name in {"back", "backward", "b"}:
        return navigator.move_backward(int(amount if amount is not None else 1))
    if name == "undo":
        navigator.undo()
        return []

    raise ValueError(f"Unknown command: {command}")


def main():
    parser = argparse.ArgumentParser(description="Run a local VMem revisit test without Gradio.")
    parser.add_argument("--config", default="configs/inference/inference.yaml")
    parser.add_argument("--image", default="test_samples/oxford.jpg")
    parser.add_argument("--out", default="revisit_outputs")
    parser.add_argument(
        "--commands",
        nargs="+",
        default=None,
        help="Sequence such as: yaw:10 yaw:-10 or left:10 right:10",
    )
    parser.add_argument(
        "--preset",
        choices=["yaw-return", "wide-yaw-return", "forward-back", "box-return"],
        default="yaw-return",
        help="Command preset used when --commands is not provided.",
    )
    parser.add_argument(
        "--auto-occlude-yaw",
        action="store_true",
        help="Use a yaw angle beyond half horizontal FOV, then return to the original view.",
    )
    parser.add_argument(
        "--occlude-margin",
        type=float,
        default=18.0,
        help="Degrees added beyond half horizontal FOV for --auto-occlude-yaw.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Repeat the command sequence this many times.",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--interp-frames", type=int, default=12)
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=0,
        help="Duplicate endpoint frames in the exported video for easier visual inspection.",
    )
    args = parser.parse_args()

    from diffusers.utils import export_to_video

    from modeling.pipeline import VMemPipeline
    from navigation import Navigator
    from utils import get_default_intrinsics, load_img_and_K, tensor_to_pil, transform_img_and_K

    os.chdir(ROOT)
    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    config.inference.visualize = False
    config.inference.visualize_pointcloud = False
    config.inference.visualize_surfel = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[revisit] device={device} image={args.image}", flush=True)

    model = VMemPipeline(config, device)
    navigator = Navigator(
        model,
        step_size=args.step_size,
        num_interpolation_frames=args.interp_frames,
    )

    image, _ = load_img_and_K(args.image, None, K=None, device=device)
    image, _ = transform_img_and_K(
        image,
        (config.model.height, config.model.width),
        mode="crop",
        K=None,
    )
    initial_frame = tensor_to_pil(image.detach().cpu())
    initial_pose = np.eye(4, dtype=np.float32)
    initial_K = np.array(get_default_intrinsics()[0], dtype=np.float32)
    hfov_deg = fov_degrees_from_intrinsics(initial_K)

    if args.commands is None:
        if args.auto_occlude_yaw:
            yaw_deg = (hfov_deg / 2.0) + args.occlude_margin
            args.commands = [f"yaw:{yaw_deg:.3f}", f"yaw:{-yaw_deg:.3f}"]
        else:
            args.commands = make_preset_commands(args.preset, hfov_deg, args.occlude_margin)
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")

    commands = args.commands * args.cycles

    print(
        f"[revisit] horizontal_fov_deg={hfov_deg:.3f} "
        f"half_fov_deg={hfov_deg / 2.0:.3f}",
        flush=True,
    )
    print(
        f"[revisit] base_commands={' '.join(args.commands)} cycles={args.cycles} "
        f"total_commands={len(commands)} expected_generated_frames~={len(commands) * args.interp_frames}",
        flush=True,
    )
    navigator.initialize(initial_frame, initial_pose, initial_K)

    all_frames = [initial_frame]
    video_frames = [initial_frame] * (args.hold_frames + 1)
    for step_idx, command in enumerate(commands, start=1):
        print(f"[revisit] step={step_idx}/{len(commands)}", flush=True)
        before_context_count = len(model.debug_context_history)
        new_frames = run_command(navigator, command) or []
        all_frames.extend(new_frames)
        video_frames.extend(new_frames)
        if args.hold_frames > 0 and video_frames:
            video_frames.extend([video_frames[-1]] * args.hold_frames)
        new_context_entries = model.debug_context_history[before_context_count:]
        for entry in new_context_entries:
            print(
                "[revisit] memory "
                f"command={command} reason={entry['reason']} "
                f"context={entry['context_time_indices']} "
                f"targets={entry['target_time_indices']}",
                flush=True,
            )
        print(f"[revisit] total_frames={len(all_frames)} surfels={len(model.surfels)}", flush=True)

    final_pose = np.asarray(navigator.current_pose)
    rotation_error = rotation_error_degrees(initial_pose, final_pose)
    translation_error = float(np.linalg.norm(final_pose[:3, 3] - initial_pose[:3, 3]))
    revisit_metrics = image_metrics(initial_frame, all_frames[-1])
    print(
        f"[revisit] final_pose_error rotation_deg={rotation_error:.6f} "
        f"translation={translation_error:.6f}",
        flush=True,
    )
    print(
        "[revisit] initial_vs_final "
        f"psnr={revisit_metrics['psnr']:.4f} "
        f"mae={revisit_metrics['mae']:.6f} "
        f"mse={revisit_metrics['mse']:.6f}",
        flush=True,
    )

    for idx, frame in enumerate(all_frames):
        frame.save(frames_dir / f"frame_{idx:04d}.png")

    video_path = out_dir / "revisit.mp4"
    export_to_video(video_frames, str(video_path), fps=args.fps)

    pose_path = out_dir / "transforms.json"
    navigator.save_camera_poses(str(pose_path))

    dense_pose_path = out_dir / "dense_transforms.json"
    save_dense_camera_poses(model.c2ws, dense_pose_path)

    memory_trace_path = out_dir / "memory_trace.json"
    with open(memory_trace_path, "w") as f:
        json.dump(model.debug_context_history, f, indent=2)

    report_path = out_dir / "report.json"
    report = {
        "image": args.image,
        "fps": args.fps,
        "interp_frames": args.interp_frames,
        "hold_frames": args.hold_frames,
        "step_size": args.step_size,
        "commands": commands,
        "num_frames": len(all_frames),
        "num_video_frames": len(video_frames),
        "num_surfels": len(model.surfels),
        "final_pose_error": {
            "rotation_deg": rotation_error,
            "translation": translation_error,
        },
        "initial_vs_final": revisit_metrics,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[revisit] saved_frames={frames_dir}", flush=True)
    print(f"[revisit] saved_video={video_path}", flush=True)
    print(f"[revisit] saved_poses={pose_path}", flush=True)
    print(f"[revisit] saved_dense_poses={dense_pose_path}", flush=True)
    print(f"[revisit] saved_memory_trace={memory_trace_path}", flush=True)
    print(f"[revisit] saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()

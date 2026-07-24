from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TRAJECTORIES = (
    "exact_revisit",
    "novel_angle_revisit",
    "partial_overlap",
    "rotation_accumulation",
    "revisit_gap",
    "contamination_followup",
    "yaw_cycle",
    "long_reverse_cycle",
)
ROTATION_SCHEDULES = {
    "90x1": (90.0, 1),
    "45x2": (45.0, 2),
    "30x3": (30.0, 3),
    "15x6": (15.0, 6),
}
GAP_REPETITIONS = {"short": 0, "medium": 2, "long": 6}


def ensure_within_repo(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    root = repo_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path must stay inside the repository: {resolved}")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_trajectory(
    name: str,
    movement_steps: int,
    yaw_step: float,
    revisit_gap: str,
    revisit_yaw_offset: float,
    rotation_schedule: str,
    cycle_yaw_steps: int = 9,
    cycle_turn_degrees: float = 90.0,
) -> list[dict]:
    if name not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory {name!r}")
    if movement_steps < 1:
        raise ValueError("movement_steps must be at least 1")
    if revisit_gap not in GAP_REPETITIONS:
        raise ValueError(f"Unknown revisit gap {revisit_gap!r}")
    if rotation_schedule not in ROTATION_SCHEDULES:
        raise ValueError(f"Unknown rotation schedule {rotation_schedule!r}")
    if cycle_yaw_steps < 1:
        raise ValueError("cycle_yaw_steps must be at least 1")
    if yaw_step == 0:
        raise ValueError("yaw_step must be non-zero")
    if cycle_turn_degrees <= 0:
        raise ValueError("cycle_turn_degrees must be positive")

    commands: list[dict] = []

    def add(command: str, phase: str, label: str) -> None:
        commands.append(
            {
                "command_index": len(commands),
                "command": command,
                "phase": phase,
                "label": label,
            }
        )

    if name in {
        "exact_revisit",
        "novel_angle_revisit",
        "partial_overlap",
        "revisit_gap",
        "contamination_followup",
    }:
        for _ in range(movement_steps):
            add("forward:1", "outbound", "move away from initial pose")

        gap_count = GAP_REPETITIONS[revisit_gap] if name == "revisit_gap" else 0
        for cycle in range(gap_count):
            add(f"yaw:{yaw_step:g}", "gap", f"gap yaw left {cycle + 1}")
            add(f"yaw:{-yaw_step:g}", "gap", f"gap yaw right {cycle + 1}")

        for _ in range(movement_steps):
            add("backward:1", "revisit", "return along the outbound path")

        if name == "novel_angle_revisit":
            add(
                f"yaw:{revisit_yaw_offset:g}",
                "revisit_offset",
                "same position with novel yaw",
            )
        elif name == "partial_overlap":
            offset = max(abs(revisit_yaw_offset), 45.0)
            add(f"yaw:{offset:g}", "revisit_offset", "partial-overlap yaw")
        elif name == "contamination_followup":
            add(f"yaw:{yaw_step:g}", "followup", "post-intervention yaw")
            add(f"yaw:{-yaw_step:g}", "followup", "post-intervention return")
    elif name == "rotation_accumulation":
        degrees, count = ROTATION_SCHEDULES[rotation_schedule]
        for step in range(count):
            add(
                f"yaw:{degrees:g}",
                "rotation_accumulation",
                f"rotation partition {step + 1}/{count}",
            )
    elif name == "yaw_cycle":
        for step in range(cycle_yaw_steps):
            add(
                f"yaw:{yaw_step:g}",
                "outbound",
                f"yaw-cycle outbound {step + 1}/{cycle_yaw_steps}",
            )
        for step in range(cycle_yaw_steps):
            add(
                f"yaw:{-yaw_step:g}",
                "revisit",
                f"yaw-cycle return {step + 1}/{cycle_yaw_steps}",
            )
    else:
        turn_steps = max(1, int(math.ceil(cycle_turn_degrees / abs(yaw_step))))
        signed_turn_step = cycle_turn_degrees / turn_steps
        outbound: list[str] = []
        outbound.extend(["forward:1"] * movement_steps)
        outbound.extend([f"yaw:{signed_turn_step:g}"] * turn_steps)
        outbound.extend(["forward:1"] * movement_steps)
        outbound.extend([f"yaw:{-signed_turn_step:g}"] * turn_steps)
        outbound.extend(["forward:1"] * movement_steps)

        for step, command in enumerate(outbound, start=1):
            add(command, "outbound", f"long-cycle outbound {step}/{len(outbound)}")

        for step, command in enumerate(reversed(outbound), start=1):
            command_name, command_value = parse_command(command)
            if command_name == "forward":
                inverse = "backward:1"
            elif command_name == "backward":
                inverse = "forward:1"
            elif command_name == "yaw" and command_value is not None:
                inverse = f"yaw:{-command_value:g}"
            else:
                raise ValueError(f"Cannot invert long-cycle command {command!r}")
            add(inverse, "revisit", f"long-cycle return {step}/{len(outbound)}")

    return commands


def parse_command(command: str) -> tuple[str, float | None]:
    name, _, raw_value = command.partition(":")
    name = name.strip().lower()
    raw_value = raw_value.strip()
    return name, float(raw_value) if raw_value else None


def rotation_error_degrees(reference_pose: np.ndarray, current_pose: np.ndarray) -> float:
    relative_rotation = reference_pose[:3, :3].T @ current_pose[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def image_metrics(reference: Image.Image, candidate: Image.Image) -> dict:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    cur = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    if ref.shape != cur.shape:
        candidate = candidate.resize(reference.size, Image.Resampling.BICUBIC)
        cur = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    difference = ref - cur
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def frame_validity_metrics(frame: Image.Image, previous: Image.Image | None = None) -> dict:
    values = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    finite = np.isfinite(values)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    saturation_ratio = float(np.mean((values <= 0.01) | (values >= 0.99)))
    result = {
        "finite": bool(finite.all()),
        "mean": mean,
        "std": std,
        "saturation_ratio": saturation_ratio,
        "black_candidate": bool(mean < 0.03 or std < 0.01),
        "saturation_candidate": bool(saturation_ratio > 0.95),
        "previous_mae": None,
        "abrupt_change_candidate": False,
        "near_duplicate_candidate": False,
    }
    if previous is not None:
        pair = image_metrics(previous, frame)
        result["previous_mae"] = pair["mae"]
        result["abrupt_change_candidate"] = bool(pair["mae"] > 0.25)
        result["near_duplicate_candidate"] = bool(pair["mse"] < 1e-6)
    return result


def save_json(path: Path, payload) -> None:
    payload = json_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def save_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    json_safe(record), ensure_ascii=False, allow_nan=False
                )
                + "\n"
            )


def json_safe(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_summary_csv(path: Path, summary: dict) -> None:
    flattened = {
        "scene": summary.get("scene"),
        "seed": summary.get("seed"),
        "context_mode": summary.get("context_mode"),
        "trajectory": summary.get("trajectory"),
        "memory_intervention": summary.get("memory_intervention"),
        "intervention_components": summary.get("intervention_components"),
        "wrong_slot_count": summary.get("wrong_slot_count"),
        "num_commands": summary.get("num_commands"),
        "generation_calls": summary.get("generation_calls"),
        "num_frames": summary.get("num_frames"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "gpu_peak_memory_mb": summary.get("gpu_peak_memory_mb"),
        "rotation_error_degrees": summary.get("final_pose_error", {}).get(
            "rotation_degrees"
        ),
        "translation_error": summary.get("final_pose_error", {}).get("translation"),
        "revisit_psnr": summary.get("revisit_consistency", {}).get("psnr"),
        "revisit_mae": summary.get("revisit_consistency", {}).get("mae"),
        "fallback_count": summary.get("retrieval", {}).get("fallback_count"),
        "heuristic_failure_frames": len(
            summary.get("rollout_validity", {}).get("flagged_frames", [])
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)


def save_manual_manifest(path: Path, frame_count: int) -> None:
    fields = (
        "frame_index",
        "failure",
        "failure_type",
        "structure_collapse",
        "repeated_texture",
        "memory_contamination",
        "notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_index in range(frame_count):
            writer.writerow({"frame_index": frame_index})


def make_contact_sheet(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    output_path: Path,
    columns: int = 4,
    thumb_size: tuple[int, int] = (256, 256),
) -> None:
    if not images:
        return
    columns = max(1, columns)
    rows = int(math.ceil(len(images) / columns))
    label_height = 28
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, columns)
        thumbnail = image.convert("RGB").copy()
        thumbnail.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = column * thumb_size[0] + (thumb_size[0] - thumbnail.width) // 2
        y = row * (thumb_size[1] + label_height)
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (column * thumb_size[0] + 4, y + thumb_size[1] + 4),
            label[:48],
            fill="black",
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)

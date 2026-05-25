#!/usr/bin/env python3
"""Local deterministic Loopwyrm snake packet preview.

This mirrors the Modal CPU smoke so the control contract can be tested without GPU.
It does not run FluxRT model inference. It intentionally uses only Python stdlib + ffmpeg
so it can run on the lean Hermes host without installing OpenCV/Pillow/Numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def clamp(v: float) -> int:
    return max(0, min(255, int(v)))


def draw_disc(buf: bytearray, width: int, height: int, cx: int, cy: int, r: int, rgb: tuple[int, int, int]) -> None:
    r2 = r * r
    for y in range(max(0, cy - r), min(height, cy + r + 1)):
        dy2 = (y - cy) * (y - cy)
        row = y * width * 3
        for x in range(max(0, cx - r), min(width, cx + r + 1)):
            if (x - cx) * (x - cx) + dy2 <= r2:
                idx = row + x * 3
                # buf is RGB
                buf[idx] = rgb[0]
                buf[idx + 1] = rgb[1]
                buf[idx + 2] = rgb[2]


def write_ppm(path: Path, width: int, height: int, data: bytearray) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(data))


def render_frame(i: int, frames: int, width: int, height: int) -> bytearray:
    t = i / max(1, frames - 1)
    buf = bytearray(width * height * 3)
    for y in range(height):
        row = y * width * 3
        for x in range(width):
            field = (math.sin(x * 0.018 + t * 12) + math.cos(y * 0.022 - t * 9)) * 32 + 40
            idx = row + x * 3
            # RGB smoky purple/green field
            buf[idx] = clamp(field * 1.8)
            buf[idx + 1] = clamp(field * 1.2)
            buf[idx + 2] = clamp(field * 0.7)

    points: list[tuple[int, int, int]] = []
    for k in range(46):
        u = k / 45
        x = int(width * (0.12 + 0.76 * u))
        y = int(height * (0.5 + 0.22 * math.sin(2 * math.pi * (u * 2.8 - t * 1.7))))
        r = max(4, int(9 + 11 * (1 - abs(u - 0.5) * 1.6)))
        points.append((x, y, r))
    for k, (x, y, r) in enumerate(points):
        draw_disc(buf, width, height, x, y, r + 2, (0, 255, 170))
        draw_disc(buf, width, height, x, y, r, (80 + k * 5 % 150, 180 + k % 70, 30 + k * 3 % 90))
    hx, hy, hr = points[-1]
    draw_disc(buf, width, height, hx, hy, hr + 7, (30, 240, 150))
    draw_disc(buf, width, height, hx + 5, hy - 5, 3, (255, 255, 255))
    draw_disc(buf, width, height, hx + 5, hy + 5, 3, (255, 255, 255))
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/loopwyrm_snake_packet")
    ap.add_argument("--prompt", default="real emerald python made of living code, assistive HUD, cyberpunk stage visuals")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "loopwyrm_snake_packet_preview.mp4"
    packet_path = out_dir / "loopwyrm_snake_packet.json"
    receipt_path = out_dir / "receipt.json"

    with tempfile.TemporaryDirectory() as td:
        frame_dir = Path(td)
        for i in range(args.frames):
            write_ppm(frame_dir / f"frame_{i:05d}.ppm", args.width, args.height, render_frame(i, args.frames, args.width, args.height))
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(args.fps), "-i", str(frame_dir / "frame_%05d.ppm"),
                "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(mp4),
            ],
            check=True,
        )

    packet = {
        "lane": "fluxrt",
        "mode": "realtime_video_edit",
        "prompt": args.prompt,
        "negative_prompt": "blurry, dead snake, flat texture, static image, watermark",
        "style_tokens": ["real_snake", "living_code", "assistive_hud", "cyberpunk_stage", "emerald_bioluminescence"],
        "control": {"target_fps": args.fps, "resolution": {"width": args.width, "height": args.height}, "prompt_update_hz": 2},
        "claim_boundary": "Procedural preview and control packet only; not FluxRT model inference.",
    }
    packet_path.write_text(json.dumps(packet, indent=2))
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(mp4)], text=True)
    receipt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preview_mp4": str(mp4),
        "packet_path": str(packet_path),
        "ffprobe": json.loads(probe),
        "claim_boundary": packet["claim_boundary"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

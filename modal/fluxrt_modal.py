"""CPU-safe Modal scaffold for FluxRT / Loopwyrm realtime visual lane.

This file is intentionally cheap and fail-closed. It does not build CUDA images,
download FLUX weights, or start realtime show control.

Run:
    modal run modal/fluxrt_modal.py::inspect_repo
    modal run modal/fluxrt_modal.py::snake_packet_smoke

The future GPU lane lives separately in `modal/fluxrt_gpu_future.py` so CPU smokes do
not accidentally build the CUDA/PyTorch/FLUX stack.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "loopwyrm-fluxrt-realtime-snake"
RECEIPT_DIR = "/outputs/receipts"

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name("loopwyrm-fluxrt-outputs", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy", "opencv-python-headless", "pillow")
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe_receipt(**kwargs):
    payload = {"timestamp": now(), **kwargs}
    Path(RECEIPT_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(RECEIPT_DIR) / f"receipt_{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"receipt_path": str(path), **payload}


@app.function(image=cpu_image, timeout=180, volumes={"/outputs": output_volume})
def inspect_repo() -> dict:
    """Cheap Modal inspection; no GPU and no model downloads."""
    root = Path.cwd()
    files = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "configs/stream_processor_config.json",
        "configs/loopwyrm_snake_modal_config.json",
        "modal/fluxrt_modal.py",
        "modal/fluxrt_gpu_future.py",
        "docs/LOOPWYRM_REALTIME_SNAKE_BRIDGE.md",
    ]
    present = {}
    for f in files:
        p = root / f
        present[f] = {"exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}
    return json_safe_receipt(
        mode="inspect_repo",
        app=APP_NAME,
        cwd=str(root),
        files=present,
        claim_boundary="CPU inspection only; no FLUX weights downloaded and no realtime inference claimed.",
    )


@app.function(image=cpu_image, timeout=240, volumes={"/outputs": output_volume})
def snake_packet_smoke(
    prompt: str = "real bioluminescent serpent made of living code, emerald scales, assistive HUD, cyberpunk stage visuals",
    fps: int = 24,
    frames: int = 96,
    width: int = 512,
    height: int = 512,
) -> dict:
    """Generate a deterministic snake-control packet + procedural preview video.

    This proves the realtime-control contract and gives the visual lane something to
    play before model weights are cached. It is not FLUXRT model inference.
    """
    import math

    import cv2
    import numpy as np

    out_dir = Path("/outputs/snake_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "loopwyrm_snake_packet_preview.mp4"
    packet_path = out_dir / "loopwyrm_snake_packet.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4), fourcc, fps, (width, height))
    for i in range(frames):
        t = i / max(1, frames - 1)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]
        field = (np.sin(xx * 0.018 + t * 12) + np.cos(yy * 0.022 - t * 9)) * 32 + 40
        img[..., 0] = np.clip(field * 0.7, 0, 255)
        img[..., 1] = np.clip(field * 1.2, 0, 255)
        img[..., 2] = np.clip(field * 1.8, 0, 255)
        points = []
        for k in range(46):
            u = k / 45
            x = int(width * (0.12 + 0.76 * u))
            y = int(height * (0.5 + 0.22 * math.sin(2 * math.pi * (u * 2.8 - t * 1.7))))
            r = int(9 + 11 * (1 - abs(u - 0.5) * 1.6))
            points.append((x, y, max(4, r)))
        for k, (x, y, r) in enumerate(points):
            color = (30 + k * 3 % 90, 180 + k % 70, 80 + k * 5 % 150)
            cv2.circle(img, (x, y), r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y), r + 2, (0, 255, 170), 1, lineType=cv2.LINE_AA)
        hx, hy, hr = points[-1]
        cv2.circle(img, (hx, hy), hr + 6, (30, 240, 150), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (hx + 5, hy - 5), 3, (255, 255, 255), -1)
        cv2.circle(img, (hx + 5, hy + 5), 3, (255, 255, 255), -1)
        cv2.putText(img, "LOOPWYRM FLUXRT DRY-RUN", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 255, 210), 2)
        cv2.putText(img, "prompt packet -> future FLUX realtime stream", (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 200, 255), 1)
        writer.write(img)
    writer.release()

    packet = {
        "lane": "fluxrt",
        "mode": "realtime_video_edit",
        "prompt": prompt,
        "negative_prompt": "blurry, dead snake, flat texture, static image, watermark",
        "style_tokens": ["real_snake", "living_code", "assistive_hud", "cyberpunk_stage", "emerald_bioluminescence"],
        "control": {
            "target_fps": fps,
            "resolution": {"width": width, "height": height},
            "prompt_update_hz": 2,
            "reference_image_enabled": True,
            "spout_or_virtualcam_output": "future_operator_machine",
            "modal_endpoint": "future_fluxrt_gpu_future_or_web_endpoint",
        },
        "claim_boundary": "Procedural preview and control packet only; not FLUXRT model inference.",
    }
    packet_path.write_text(json.dumps(packet, indent=2))
    ffprobe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(mp4)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    ).stdout
    return json_safe_receipt(
        mode="snake_packet_smoke",
        packet_path=str(packet_path),
        preview_mp4=str(mp4),
        prompt=prompt,
        ffprobe=json.loads(ffprobe or "{}"),
        claim_boundary="Procedural preview + control packet only; no model weights or realtime FLUX inference used.",
    )

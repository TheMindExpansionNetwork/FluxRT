"""Future heavy GPU scaffold for FluxRT on Modal.

Do not use this for cheap smokes. It intentionally lives in a separate file from
`fluxrt_modal.py` because Modal builds every image referenced by a file during a
run; separating the CUDA lane prevents accidental multi-GB PyTorch builds when we
only need a CPU control-packet receipt.

Future preflight after model cache is intentionally populated:
    modal run modal/fluxrt_gpu_future.py::fluxrt_video_smoke

Future real smoke after wiring StreamProcessor and cached weights:
    modal run modal/fluxrt_gpu_future.py::fluxrt_video_smoke --run-real
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "loopwyrm-fluxrt-realtime-snake-gpu-future"
CACHE_DIR = "/cache/fluxrt"
RECEIPT_DIR = "/outputs/receipts"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("loopwyrm-fluxrt-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("loopwyrm-fluxrt-outputs", create_if_missing=True)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements("requirements.txt")
)


def json_safe_receipt(**kwargs):
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **kwargs}
    Path(RECEIPT_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(RECEIPT_DIR) / f"gpu_future_receipt_{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"receipt_path": str(path), **payload}


@app.function(
    image=gpu_image,
    gpu="L40S",
    timeout=900,
    volumes={CACHE_DIR: cache_volume, "/outputs": output_volume},
)
def fluxrt_video_smoke(
    run_real: bool = False,
    prompt: str = "turn this into a real emerald snake made of living code, assistive video HUD, cyberpunk lighting",
    seconds: int = 4,
) -> dict:
    """Future bounded GPU smoke, fail-closed until weights are cached and real mode is explicit."""
    model_path = Path(CACHE_DIR) / "FLUX.2-klein-4B"
    rife_path = Path(CACHE_DIR) / "RIFE-safetensors"
    preflight = {
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "rife_path": str(rife_path),
        "rife_exists": rife_path.exists(),
        "run_real": run_real,
        "seconds": seconds,
        "prompt": prompt,
    }
    if not run_real:
        return json_safe_receipt(
            mode="gpu_preflight_refused_by_default",
            preflight=preflight,
            next_step="Cache FLUX.2-klein-4B and RIFE into the Modal volume, then rerun with --run-real.",
            claim_boundary="GPU image/endpoint scaffold only; no realtime inference claimed.",
        )
    if not (model_path.exists() and rife_path.exists()):
        return json_safe_receipt(
            mode="gpu_preflight_missing_models",
            preflight=preflight,
            error="Required model directories not found in Modal volume; refusing hidden downloads.",
            claim_boundary="No inference run; missing cached weights.",
        )
    return json_safe_receipt(
        mode="gpu_ready_placeholder",
        preflight=preflight,
        next_step="Wire FluxRT StreamProcessor to a synthetic input clip and write /outputs/fluxrt_real_snake_smoke.mp4.",
        claim_boundary="Model paths exist but real inference has not been wired in this scaffold.",
    )

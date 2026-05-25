# Loopwyrm FluxRT realtime snake bridge

Status: `modal_scaffold_dry_run_first`

This fork turns FluxRT into a Loopwyrm visual lane for realtime assistive video/code visuals: living-code snakes, HUD overlays, prompt-swaps, webcam/video stream edits, and later TouchDesigner/Resolume/OBS routing.

## Honest boundary

What is ready in this increment:

- FluxRT is forked locally under TheMindExpansionNetwork.
- A Modal app scaffold exists at `modal/fluxrt_modal.py`.
- CPU-safe checks can run without downloading weights.
- A deterministic snake-control packet + procedural MP4 preview can be generated.
- GPU function is fail-closed until FLUX/RIFE model directories are intentionally cached.

What is **not** claimed yet:

- No FLUX.2-klein-4B weights downloaded by default.
- No live webcam/virtualcam/Spout/Resolume output from Modal yet.
- No TouchDesigner MCP control from this headless Linux agent.
- No true realtime FLUX inference receipt until GPU smoke produces timed frames/video.

## Why Modal

FluxRT wants CUDA 12.8, Python 3.12, PyTorch CUDA, and 20GB+ VRAM. Modal is the right cloud lane for bounded GPU tests without turning the local machine into a fragile show box.

The Modal files split cheap and heavy work:

- `modal/fluxrt_modal.py::inspect_repo` — CPU Modal inspection.
- `modal/fluxrt_modal.py::snake_packet_smoke` — CPU deterministic control-packet + procedural snake preview.
- `modal/fluxrt_gpu_future.py::fluxrt_video_smoke` — future L40S GPU lane, refuses hidden model downloads unless `run_real=True` and cached model dirs exist.

The GPU lane is in a **separate file** because Modal builds every image referenced by a run file; this prevents accidental CUDA/PyTorch builds when we only need a cheap CPU receipt.

## Commands

From the repo root:

```bash
modal run modal/fluxrt_modal.py::inspect_repo
modal run modal/fluxrt_modal.py::snake_packet_smoke
```

Future gated GPU preflight:

```bash
modal run modal/fluxrt_gpu_future.py::fluxrt_video_smoke
```

Future real run, only after model cache is intentionally populated:

```bash
modal run modal/fluxrt_gpu_future.py::fluxrt_video_smoke \
  --run-real \
  --prompt "turn the live stream into a real emerald python made of living code, scales reflecting terminal text, assistive HUD, cyberpunk stage lighting"
```


## Audio spine bridge

Canonical audio spine repo:

```text
https://github.com/TheMindExpansionNetwork/loopwyrm_endless_jam_spine
```

The spine is the source-of-truth contract for continuous Loopwyrm/Sonic-Forage audio:

```text
SA3 / ACE-Step / Dasheng / future LMDM workers
-> loop_bank/<lane>/incoming/*.wav|*.mp3
-> normalizer writes loop_bank/<lane>/ready/*__normalized.*
-> mixer/crossfader writes loop_bank/mixed/exports/*.mp3
-> receipt JSON records clips_used, hashes, durations, and claim boundaries
```

Verified in this integration pass using uv-managed NumPy (the host `python3` lacked NumPy):

```bash
cd /opt/data/workspace/projects/loopwyrm_endless_jam_spine
uv run --with numpy python scripts/run_endless_spine.py \
  --root loop_bank \
  --duration-min 1 \
  --out loop_bank/mixed/exports/loopwyrm_endless_jam_spine_verify_1min.mp3 \
  --demo-seed-if-empty
```

Receipt/proof copied into the katalog repo:

```text
loop_bank/mixed/exports/loopwyrm_endless_jam_spine_verify_1min.mp3
receipts/LATEST_LOOPWYRM_SPINE_FLUXRT_MAPPING.json
```

Proof SHA-256: `11fc0917da43c79b0d6759f14b45c16c539a841a393800ab385cf8169e943fb5`

FluxRT should consume this as a timeline/control-sync signal, not as proof of live video inference. The current FluxRT lane remains: procedural control packet + Modal CPU scaffold only; GPU FLUX/RIFE inference is still fail-closed until model caches and a GPU receipt exist.

## Intended live architecture

```text
Hermes / operator prompt / code changes
        |
        v
Loopwyrm control packet
        |
        +--> FluxRT Modal lane: realtime video edit frames
        +--> StreamDiffusionV2 lane: streaming video generation experiments
        +--> TouchDiffusion/TouchDesigner lane: local VJ composition
        +--> Resolume/OBS lane: operator-machine output
        |
        v
folder-backed preview/export queue + websocket/control API
```

## Snake pattern control packet

Example packet emitted by `snake_packet_smoke`:

```json
{
  "lane": "fluxrt",
  "mode": "realtime_video_edit",
  "prompt": "real bioluminescent serpent made of living code, emerald scales, assistive HUD, cyberpunk stage visuals",
  "negative_prompt": "blurry, dead snake, flat texture, static image, watermark",
  "style_tokens": ["real_snake", "living_code", "assistive_hud", "cyberpunk_stage", "emerald_bioluminescence"],
  "control": {
    "target_fps": 24,
    "resolution": {"width": 512, "height": 512},
    "prompt_update_hz": 2,
    "reference_image_enabled": true,
    "spout_or_virtualcam_output": "future_operator_machine"
  }
}
```

## Local show-control plan

Because the user has Resolume and may use TouchDesigner later:

1. Modal generates edited preview frames/video or streams endpoint frames.
2. Local operator machine receives frames through a small bridge app.
3. TouchDesigner/Resolume handles projection/VJ output, Spout/Syphon/NDI/virtual camera, and live compositing.
4. Hermes modifies prompt/control packets in realtime: “more real snake,” “snake becomes code,” “wrap around performer,” “turn into glowing python scales,” etc.

Do not start unattended live show control. Keep it fail-closed until the operator confirms the local machine, GPU route, and display outputs.

## Model cache notes

Required upstream model folders from README:

```text
RIFE-safetensors/
FLUX.2-klein-4B/
FLUX.2-klein-4B-int8/ optional
```

Recommended Modal cache volume:

```text
loopwyrm-fluxrt-cache
```

Keep large model weights out of git. Commit only scripts, configs, docs, and receipts.

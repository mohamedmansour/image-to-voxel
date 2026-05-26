# Copilot Instructions for Image-to-Voxel Pipeline

## Build, test, and lint commands

This directory does not define a formal build system, linter, or automated test suite. Use the pipeline scripts directly.

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git triposr_repo
pip install git+https://github.com/tatsy/torchmcubes.git
pip install omegaconf einops rtree xatlas moderngl imageio
```

```bash
# Main pipeline (single asset)
source .venv/bin/activate
python -m src.pipeline --emoji "🐶" --resolution 32
```

```bash
# "Single test" style stage-level smoke run
source .venv/bin/activate
python -m src.voxelize mesh.obj --resolution 32 --output voxels.json --name "sample"
```

## High-level architecture

The pipeline converts 2D input (emoji/image) into web-ready voxel atlas data through four scripts:

1. `prepare_image.py`: renders emoji or loads image, optionally removes background, applies stylization cues for better reconstruction.
2. `image_to_mesh.py`: runs TripoSR (prefers MPS on Apple Silicon, falls back to CPU) and exports colored mesh (`.obj`/`.glb`).
3. `voxelize.py`: normalizes mesh to unit cube, voxelizes, keeps surface shell, samples mesh colors, writes compact voxel JSON.
4. `export_voxels.py`: merges per-asset JSONs into one `voxel_atlas.json`.

`pipeline.py` orchestrates single-item and batch execution. Batch mode is a concurrent 3-stage pipeline:

- Stage 1 CPU process pool: image preparation
- Stage 2 GPU thread: TripoSR inference (batched, size 4)
- Stage 3 CPU process pool: voxelization

Outputs are written under `output/` and atlas output is copied to `../../../public/voxel_assets/voxel_atlas.json` for game loading.

## Key conventions in this codebase

- Emoji catalogs in `pipeline.py` are the source of truth for batch generation (`GAME_EMOJIS`).
- Emoji file naming must use `emoji_to_filename()` (codepoint-based safe names), producing files like `<safe>_prepared.png`, `<safe>.obj`, `<safe>_voxels.json`.
- Batch resume semantics are file-based: existing `<safe>_voxels.json` means "already processed" unless `--no-resume` is set.
- Multiprocessing helpers used by `ProcessPoolExecutor` are top-level functions (`_prepare_one`, `_voxelize_one`) and should remain picklable.
- Atlas format is intentionally compact flat arrays (`positions`, `colors`), not per-voxel object arrays; keep this shape stable for downstream loaders.
- In image mode, background removal is on by default; in emoji mode it is skipped by default.

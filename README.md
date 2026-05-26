# Image-to-Voxel Pipeline

Converts any image (or emoji) into a 3D voxelized mesh for use in the Smash game. Runs locally on Apple Silicon GPU (MPS) using [TripoSR](https://huggingface.co/stabilityai/TripoSR) from Hugging Face.

## Setup (one-time)

```bash
cd scripts/image_to_voxel

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Clone TripoSR (the 3D reconstruction model)
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git triposr_repo

# Install TripoSR-specific dependencies
pip install git+https://github.com/tatsy/torchmcubes.git
pip install omegaconf einops rtree xatlas moderngl imageio
```

The TripoSR model weights (~1.7 GB) are downloaded automatically from Hugging Face on first run.

## Usage

Always activate the venv first:

```bash
cd scripts/image_to_voxel
source .venv/bin/activate
```

### Single emoji

```bash
python pipeline.py --emoji "🐶" --resolution 32
```

### Single image (any PNG/JPG)

```bash
python pipeline.py --image ./my_sprite.png --name "hero" --resolution 32
```

### Batch all game emojis

```bash
python pipeline.py --batch-emojis --resolution 32
```

### Batch specific categories

```bash
python pipeline.py --batch-emojis --categories animals food --resolution 32
```

### Higher resolution

```bash
python pipeline.py --emoji "🐶" --resolution 64
```

## Re-running / Regenerating

The pipeline supports **resume by default** — it skips emojis that already have a `_voxels.json` output. Just re-run the same command and it picks up where it left off:

```bash
# Resume from where it stopped (skips already-processed)
python pipeline.py --batch-emojis --resolution 32

# Force re-process everything (no resume)
python pipeline.py --batch-emojis --resolution 32 --no-resume

# Clear all output and start fresh
rm -rf output/
python pipeline.py --batch-emojis --resolution 32
```

The pipeline automatically copies the atlas to `public/voxel_assets/voxel_atlas.json` when finished.

## Parallelism & Performance

The batch pipeline uses a **concurrent 3-stage architecture** to maximize hardware utilization on Apple Silicon:

```
Stage 1 (CPU pool)      Stage 2 (GPU)         Stage 3 (CPU pool)
Image Preparation  →→→  TripoSR Inference  →→→  Voxelization
  N workers              Batched (4/pass)        N workers
  ProcessPoolExecutor    MPS GPU                 ProcessPoolExecutor
```

- **Stage 1**: Renders emojis and applies stylization across all CPU cores via `ProcessPoolExecutor`
- **Stage 2**: Runs TripoSR on the MPS GPU, processing **4 images per forward pass** for better throughput
- **Stage 3**: Voxelizes meshes in parallel across all CPU cores via `ProcessPoolExecutor`

All 3 stages run **simultaneously** via queues — while the GPU processes batch N, CPUs are preparing batch N+1 and voxelizing batch N-1.

### Typical performance (M4 Pro, 48 GB)

- **~5 seconds per emoji** (GPU-bottlenecked)
- **~225 emojis in ~19 minutes**
- GPU utilization: ~80%, CPU: ~60%, RAM: ~23 GB

The GPU is the bottleneck — TripoSR inference is sequential. CPU stages overlap with GPU work so no time is wasted.

### Live progress bar

Batch runs show a single-line progress bar with all 3 stages:

```
  ████████░░░░░░░░░░░░  40% │ 📋 225/225 → 🖥️  100/225 → 🧊 90/225 │ ⏱️  450s (ETA 675s) │ 🐶
```

## Loading in the Game

The game loads voxel assets automatically at startup:

1. `src/voxelLoader.ts` fetches `/voxel_assets/voxel_atlas.json`
2. `src/voxelFactory.ts` checks for voxel data when creating emoji meshes
3. If voxel data exists → renders as 3D voxel mesh with vertex colors + flat shading
4. If not → falls back to the flat textured cube

### Dev server

The dev server (`server.ts`) uses `Bun.serve()` with HTML imports and serves `public/` as static assets:

```bash
pnpm dev   # runs bun --hot server.ts on port 3000
```

For production, deploy `public/voxel_assets/` to your static host. The loader fetches from `/voxel_assets/voxel_atlas.json` by default — pass a custom URL to `preloadVoxelAtlas()` if hosted elsewhere.

## Pipeline Stages

```
Image → Prepare+Stylize → TripoSR (Image-to-3D) → Voxelize → Export JSON
```

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `prepare_image.py` | Render emoji or load image, apply shading/AO for depth cues |
| 2 | `image_to_mesh.py` | TripoSR inference on MPS GPU → triangle mesh with vertex colors |
| 3 | `voxelize.py` | Mesh → voxel grid at target resolution, surface color sampling |
| 4 | `export_voxels.py` | Combine per-asset JSONs into a single atlas |

### Running stages individually

```bash
# Stage 1: Prepare image
python prepare_image.py --emoji "🍕" --output prepared.png

# Stage 2: Image to mesh
python image_to_mesh.py prepared.png --output mesh.obj

# Stage 3: Voxelize mesh
python voxelize.py mesh.obj --resolution 32 --output voxels.json --name "🍕"

# Stage 4: Build atlas from all individual JSONs
python export_voxels.py output/ --output ../../public/voxel_assets/voxel_atlas.json
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--emoji` | Emoji character to process | — |
| `--image` | Path to input image (PNG/JPG) | — |
| `--batch-emojis` | Process all game emojis | — |
| `--name` | Asset name (for `--image` mode) | filename |
| `--resolution`, `-r` | Voxel grid resolution | 32 |
| `--mc-resolution` | Marching cubes resolution for mesh extraction | 256 |
| `--output-dir`, `-o` | Output directory | `output` |
| `--categories` | Emoji categories: `animals`, `transport`, `food`, `faces` | all |
| `--no-resume` | Don't skip already-processed emojis | off |
| `--cpu` | Force CPU (no GPU) | off |
| `--skip-rembg` | Skip background removal (for `--image` mode) | off |

## System Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4) — uses MPS for GPU acceleration
- **Python 3.9+**
- **~6 GB GPU memory** for TripoSR model
- **~23 GB RAM** during batch processing (parallel workers)
- **~5 seconds per emoji** on M4 Pro

## Output Format

The atlas JSON uses compact flat arrays for efficient loading:

```json
{
  "🐶": {
    "resolution": 32,
    "count": 1234,
    "bounds": { "min": [0, 0, 0], "max": [31, 31, 31] },
    "positions": [x0, y0, z0, x1, y1, z1, ...],
    "colors": [r0, g0, b0, r1, g1, b1, ...]
  }
}
```

The Three.js integration (`voxelFactory.ts`) builds `BufferGeometry` with:
- **Exposed-face-only rendering** — adjacent voxels share faces, only visible faces are emitted
- **Vertex colors** — per-voxel RGB from the mesh surface
- **Flat shading** — matches the Minecraft-like letter voxel aesthetic

## Troubleshooting

**Blank/identical meshes**: The emoji font rendering failed. Apple Color Emoji only supports specific pixel sizes (160, 96, etc.). Verify with:
```bash
python -c "from prepare_image import render_emoji; import numpy as np; img = render_emoji('🍕'); print('std:', np.array(img).std())"
```
`std` should be > 0. If 0, the emoji rendered as blank white.

**MPS errors**: TripoSR auto-falls back to CPU for marching cubes (torchmcubes doesn't support Metal). If the full model fails on MPS, use `--cpu`.

**Out of memory**: Reduce `--mc-resolution` (default 256, try 128) or reduce parallel workers.

**Resume not working**: The pipeline checks for `<name>_voxels.json` in the output dir. If you changed resolution or want to regenerate, use `--no-resume`.

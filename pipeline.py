"""
Image-to-Voxel Pipeline CLI Orchestrator.

Converts any image (or emoji) to a 3D voxelized mesh through:
  1. Image preparation & stylization
  2. Image-to-3D mesh (TripoSR)
  3. Mesh voxelization
  4. Export to compact JSON

Usage:
  python pipeline.py --emoji "🐶" --resolution 32
  python pipeline.py --image ./sprite.png --name "hero" --resolution 32
  python pipeline.py --batch-emojis --resolution 32
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict

# All emojis from the game's modes.ts
GAME_EMOJIS = {
    "animals": [
        "🐶","🐱","🐭","🐹","🐰","🦊","🐻","🐼","🐨","🐯",
        "🦁","🐮","🐷","🐸","🐵","🐔","🐧","🐦","🦆","🐴",
        "🦄","🐝","🦋","🐢","🐍","🐙","🐠","🐬","🐳","🦈",
        "🐊","🐘","🦒","🦘","🦩","🦜","🐿️","🦔","🦥","🐓",
        "🦝","🦡","🦫","🦃","🦤","🦚","🦢","🐇","🐁","🐀",
        "🦨","🦦","🐕","🐈","🐅","🐆","🦓","🦍","🦧","🐪",
        "🐫","🦙","🦣","🐃","🐂","🐄","🐎","🐖","🐐","🐑","🐏",
    ],
    "transport": [
        "🚗","🚕","🚙","🚌","🏎️","🚓","🚑","🚒","🚐","🚚",
        "🚜","🏍️","🛵","🚲","🚂","🚄","🚅","🚇","✈️","🚀",
        "🛸","🚁","⛵","🚤","🚢","🛶","🚠","🛻","🚃","🚡",
        "🛺","🏗️","🚛","🚎","🛩️","🚟","🛤️","⛽","🚏","🛣️",
        "🚧","⚓","🛥️","🚿","🛞","🛴","🛳️","🚞",
    ],
    "food": [
        "🍕","🍔","🍟","🌭","🍿","🧁","🍩","🍪","🎂","🍰",
        "🍫","🍬","🍭","🍯","🍎","🍊","🍋","🍌","🍉","🍇",
        "🍓","🫐","🍑","🥝","🍍","🥭","🥑","🧀","🥐",
        "🥨","🥯","🍖","🍗","🥩","🌯","🫕","🥗","🥘","🫔",
        "🥙","🧆","🥚","🧈","🥞","🧇","🍤","🦐","🦑","🦪",
        "🍱","🍘","🍙","🍚","🍛","🍝","🥮","🥟","🍠","🥧",
    ],
    "faces": [
        "😀","😃","😄","😁","😆","🥹","😅","😂","🤣","🥲",
        "😊","😇","🙂","🙃","😉","😌","😍","🥰","😘","😗",
        "😙","😚","😋","😛","😜","🤪","😝","🤑","🤗","🤭",
        "🫢","🫣","🤫","🤔","🫡","🤐","🤨","😐","😑","😶",
        "🫥","😏","😒","🙄","😬","🤥","😔",
    ],
}


def get_all_emojis() -> list[str]:
    """Get all emoji characters from the game."""
    emojis = []
    for category in GAME_EMOJIS.values():
        emojis.extend(category)
    return emojis


def emoji_to_filename(emoji: str) -> str:
    """Convert an emoji to a safe filename using codepoints."""
    codepoints = "-".join(f"{ord(c):x}" for c in emoji if ord(c) > 0xFF)
    return codepoints or f"u{ord(emoji[0]):04x}"


def process_single(
    emoji: str | None = None,
    image_path: str | None = None,
    name: str | None = None,
    resolution: int = 32,
    output_dir: str = "output",
    mc_resolution: int = 256,
    skip_rembg: bool = False,
    cpu: bool = False,
) -> dict | None:
    """Process a single image through the full pipeline."""
    import torch
    from prepare_image import prepare_image
    from image_to_mesh import image_to_mesh, get_device
    from voxelize import voxelize_mesh

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine asset name
    if name:
        asset_name = name
    elif emoji:
        asset_name = emoji
    elif image_path:
        asset_name = Path(image_path).stem
    else:
        raise ValueError("Must provide --emoji, --image, or --name")

    safe_name = emoji_to_filename(emoji) if emoji else Path(image_path).stem

    # Stage 1: Prepare image
    print(f"\n{'='*60}")
    print(f"Processing: {asset_name}")
    print(f"{'='*60}")

    prepared_path = out_dir / f"{safe_name}_prepared.png"
    print(f"\n[Stage 1/3] Preparing image...")
    img = prepare_image(
        emoji=emoji,
        image_path=image_path,
        output_path=str(prepared_path),
        skip_rembg=skip_rembg or bool(emoji),
    )

    # Stage 2: Image to mesh
    mesh_path = out_dir / f"{safe_name}.obj"
    print(f"\n[Stage 2/3] Converting to 3D mesh...")
    device = torch.device("cpu") if cpu else get_device()
    try:
        image_to_mesh(
            str(prepared_path),
            str(mesh_path),
            device=device,
            mc_resolution=mc_resolution,
        )
    except Exception as e:
        print(f"  ERROR in mesh generation: {e}")
        if "mps" in str(device):
            print("  Retrying with CPU fallback...")
            image_to_mesh(
                str(prepared_path),
                str(mesh_path),
                device=torch.device("cpu"),
                mc_resolution=mc_resolution,
            )
        else:
            raise

    # Stage 3: Voxelize
    voxel_path = out_dir / f"{safe_name}_voxels.json"
    print(f"\n[Stage 3/3] Voxelizing at {resolution}³...")
    try:
        result = voxelize_mesh(
            str(mesh_path),
            resolution=resolution,
            output_path=str(voxel_path),
            name=asset_name,
        )
    except Exception as e:
        print(f"  ERROR in voxelization: {e}")
        return None

    print(f"\n✅ Done: {asset_name} → {result['count']} voxels")
    return result


def batch_emojis(
    resolution: int = 32,
    output_dir: str = "output",
    mc_resolution: int = 256,
    resume: bool = True,
    cpu: bool = False,
    categories: list[str] | None = None,
) -> None:
    """Process all game emojis with a concurrent 3-stage pipeline.

    All stages run simultaneously via queues:
    - Stage 1 (CPU pool): Image preparation feeds into →
    - Stage 2 (GPU thread): TripoSR inference feeds into →
    - Stage 3 (CPU pool): Voxelization
    This keeps CPU cores AND GPU busy at all times.
    """
    import concurrent.futures
    import multiprocessing
    import queue
    import threading
    import torch
    from export_voxels import build_atlas
    from prepare_image import prepare_image
    from image_to_mesh import batch_image_to_mesh, get_device, load_triposr
    from voxelize import voxelize_mesh

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect emojis
    if categories:
        emojis = []
        for cat in categories:
            if cat in GAME_EMOJIS:
                emojis.extend(GAME_EMOJIS[cat])
            else:
                print(f"Warning: unknown category '{cat}'")
    else:
        emojis = get_all_emojis()

    # Filter already-processed
    work_items = []
    skipped = 0
    for emoji in emojis:
        safe_name = emoji_to_filename(emoji)
        voxel_file = out_dir / f"{safe_name}_voxels.json"
        if resume and voxel_file.exists():
            skipped += 1
            continue
        work_items.append((emoji, safe_name))

    total = len(work_items)
    print(f"Processing {total} emojis at {resolution}³ resolution ({skipped} already done)")
    if total == 0:
        print("Nothing to do!")
        _finalize_atlas(out_dir)
        return

    # Pre-warm the TripoSR model
    device = torch.device("cpu") if cpu else get_device()
    print("Pre-loading TripoSR model...")
    load_triposr(device)

    num_cpu_workers = max(2, multiprocessing.cpu_count() - 2)
    print(f"Pipeline: {num_cpu_workers} CPU workers + 1 GPU thread\n")

    # Queues between stages
    prep_to_gpu = queue.Queue(maxsize=32)   # prepared images → GPU (big buffer)
    gpu_to_voxel = queue.Queue(maxsize=32)  # meshes → voxelization

    GPU_BATCH_SIZE = 4  # Process multiple images per GPU forward pass

    # Shared progress counters
    progress = {"prep": 0, "gpu": 0, "voxel": 0, "failed": 0}
    start_time = time.time()
    lock = threading.Lock()

    def update_progress(stage: str, emoji: str = ""):
        with lock:
            progress[stage] += 1
            elapsed = time.time() - start_time
            p, g, v = progress["prep"], progress["gpu"], progress["voxel"]
            f = progress["failed"]
            rate = v / elapsed if elapsed > 0 and v > 0 else 0
            eta = (total - v) / rate if rate > 0 else 0
            bar_w = 20
            filled = int(bar_w * v / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = v * 100 // total if total > 0 else 0
            status = (
                f"\r  {bar} {pct:3d}% │ "
                f"📋 {p}/{total} → 🖥️  {g}/{total} → 🧊 {v}/{total} │ "
                f"{'❌ ' + str(f) + ' │ ' if f else ''}"
                f"⏱️  {elapsed:.0f}s"
            )
            if eta > 0:
                status += f" (ETA {eta:.0f}s)"
            if emoji:
                status += f" │ {emoji}"
            print(status, end="", flush=True)

    def fail_progress(emoji: str, stage: str, err: str):
        with lock:
            progress["failed"] += 1
        print(f"\n  ✗ {emoji} {stage} failed: {err}", flush=True)

    # Stage 1: Prepare images (process pool for true CPU parallelism)
    def prep_worker():
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu_workers) as pool:
            futures = {
                pool.submit(_prepare_one, emoji, safe_name, str(out_dir)): (emoji, safe_name)
                for emoji, safe_name in work_items
            }
            for f in concurrent.futures.as_completed(futures):
                emoji, safe_name, path, err = f.result()
                if err:
                    fail_progress(emoji, "prep", err)
                else:
                    update_progress("prep", emoji)
                    prep_to_gpu.put((emoji, safe_name, path))
        prep_to_gpu.put(None)

    # Stage 2: GPU inference with batching (process N images per forward pass)
    GPU_BATCH_SIZE = 4
    def gpu_worker():
        while True:
            batch = []
            item = prep_to_gpu.get()
            if item is None:
                break
            batch.append(item)
            # Fill batch from queue without blocking
            while len(batch) < GPU_BATCH_SIZE:
                try:
                    item = prep_to_gpu.get_nowait()
                    if item is None:
                        prep_to_gpu.put(None)  # re-queue sentinel
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            items_for_gpu = [
                (path, str(out_dir / f"{sn}.obj"))
                for _, sn, path in batch
            ]
            try:
                results = batch_image_to_mesh(
                    items_for_gpu, device=device, mc_resolution=mc_resolution,
                )
                for (emoji, safe_name, _), (out_path, success, err) in zip(batch, results):
                    if success:
                        update_progress("gpu", emoji)
                        gpu_to_voxel.put((emoji, safe_name, out_path))
                    else:
                        fail_progress(emoji, "mesh", err or "unknown")
            except Exception as e:
                # Fallback: process one at a time
                from image_to_mesh import image_to_mesh
                for emoji, safe_name, prepared_path in batch:
                    mesh_path = str(out_dir / f"{safe_name}.obj")
                    try:
                        image_to_mesh(prepared_path, mesh_path, device=device, mc_resolution=mc_resolution)
                        update_progress("gpu", emoji)
                        gpu_to_voxel.put((emoji, safe_name, mesh_path))
                    except Exception as e2:
                        fail_progress(emoji, "mesh", str(e2))
        gpu_to_voxel.put(None)

    # Stage 3: Voxelization
    def voxel_worker():
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu_workers) as pool:
            pending = {}
            while True:
                try:
                    item = gpu_to_voxel.get(timeout=0.5)
                    if item is None:
                        break
                    emoji, safe_name, mesh_path = item
                    future = pool.submit(
                        _voxelize_one, mesh_path, resolution,
                        str(out_dir / f"{safe_name}_voxels.json"), emoji,
                    )
                    pending[future] = emoji
                except queue.Empty:
                    pass

                done = [f for f in pending if f.done()]
                for f in done:
                    emoji = pending.pop(f)
                    try:
                        success, err = f.result()
                        if success:
                            update_progress("voxel", emoji)
                        else:
                            fail_progress(emoji, "voxelize", err)
                    except Exception as e:
                        fail_progress(emoji, "voxelize", str(e))

            for f in concurrent.futures.as_completed(pending):
                emoji = pending[f]
                try:
                    success, err = f.result()
                    if success:
                        update_progress("voxel", emoji)
                    else:
                        fail_progress(emoji, "voxelize", err)
                except Exception as e:
                    fail_progress(emoji, "voxelize", str(e))

    # Launch pipeline
    print(f"\r  {'░' * 20}   0% │ 📋 0/{total} → 🖥️  0/{total} → 🧊 0/{total} │ ⏱️  0s", end="", flush=True)
    t1 = threading.Thread(target=prep_worker, name="prep")
    t2 = threading.Thread(target=gpu_worker, name="gpu")
    t3 = threading.Thread(target=voxel_worker, name="voxel")

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    elapsed = time.time() - start_time
    done = progress["voxel"]
    failed = progress["failed"]
    print(f"\n\n{'='*60}")
    print(f"✅ Complete: {done} processed, {skipped} skipped, {failed} failed")
    print(f"⏱️  Total: {elapsed:.1f}s ({elapsed/max(done,1):.1f}s per emoji)")

    _finalize_atlas(out_dir)


def _prepare_one(emoji, safe_name, out_dir):
    """Standalone function for ProcessPoolExecutor (must be top-level picklable)."""
    try:
        from prepare_image import prepare_image
        prepared_path = str(Path(out_dir) / f"{safe_name}_prepared.png")
        prepare_image(emoji=emoji, output_path=prepared_path, skip_rembg=True)
        return (emoji, safe_name, prepared_path, None)
    except Exception as e:
        return (emoji, safe_name, None, str(e))


def _voxelize_one(mesh_path, resolution, output_path, name):
    """Standalone function for ProcessPoolExecutor (must be top-level picklable)."""
    try:
        from voxelize import voxelize_mesh
        voxelize_mesh(mesh_path, resolution=resolution, output_path=output_path, name=name)
        return (True, None)
    except Exception as e:
        return (False, str(e))


def _finalize_atlas(out_dir: Path) -> None:
    """Build atlas and copy to public directory."""
    from export_voxels import build_atlas

    atlas_path = str(out_dir / "voxel_atlas.json")
    print(f"\nBuilding atlas...")
    build_atlas(str(out_dir), atlas_path)

    # Copy to public assets dir for the game to load
    public_dir = Path(__file__).parent.parent.parent / "public" / "voxel_assets"
    public_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(atlas_path, str(public_dir / "voxel_atlas.json"))
    print(f"Copied atlas to {public_dir / 'voxel_atlas.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Image-to-Voxel Pipeline: convert images to 3D voxel assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --emoji "🐶" --resolution 32
  %(prog)s --image ./sprite.png --name "hero" --resolution 64
  %(prog)s --batch-emojis --resolution 32
  %(prog)s --batch-emojis --categories animals food --resolution 32
        """,
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--emoji", type=str, help="Single emoji character")
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--batch-emojis", action="store_true",
                             help="Process all game emojis")

    # Options
    parser.add_argument("--name", type=str, help="Asset name (for --image mode)")
    parser.add_argument("--resolution", "-r", type=int, default=32,
                        help="Voxel resolution (default: 32)")
    parser.add_argument("--mc-resolution", type=int, default=256,
                        help="Marching cubes resolution for mesh extraction (default: 256)")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--categories", nargs="+", type=str,
                        choices=list(GAME_EMOJIS.keys()),
                        help="Emoji categories to process (for --batch-emojis)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip already-processed emojis")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (no GPU acceleration)")
    parser.add_argument("--skip-rembg", action="store_true",
                        help="Skip background removal for --image input")

    args = parser.parse_args()

    if args.batch_emojis:
        batch_emojis(
            resolution=args.resolution,
            output_dir=args.output_dir,
            mc_resolution=args.mc_resolution,
            resume=not args.no_resume,
            cpu=args.cpu,
            categories=args.categories,
        )
    else:
        result = process_single(
            emoji=args.emoji,
            image_path=args.image,
            name=args.name,
            resolution=args.resolution,
            output_dir=args.output_dir,
            mc_resolution=args.mc_resolution,
            skip_rembg=args.skip_rembg,
            cpu=args.cpu,
        )
        if result:
            # For single items, also copy to the public assets dir
            public_dir = Path(__file__).parent.parent.parent / "public" / "voxel_assets"
            public_dir.mkdir(parents=True, exist_ok=True)

            from export_voxels import build_atlas
            build_atlas(args.output_dir, str(public_dir / "voxel_atlas.json"))


if __name__ == "__main__":
    main()

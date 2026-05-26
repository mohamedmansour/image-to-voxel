"""
Image-to-3D mesh conversion using TripoSR from Hugging Face.

Converts a prepared 2D image into a 3D triangle mesh with vertex colors.
Runs on MPS (Apple Silicon) with CPU fallback.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Add the local TripoSR repo to path
TRIPOSR_DIR = os.path.join(os.path.dirname(__file__), "triposr_repo")
if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING
)


def get_device() -> torch.device:
    """Get the best available device, preferring MPS for Apple Silicon."""
    if torch.backends.mps.is_available():
        logging.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logging.info("Using CUDA GPU")
        return torch.device("cuda")
    else:
        logging.info("Using CPU (no GPU acceleration)")
        return torch.device("cpu")


_model_cache = {}


def load_triposr(device: torch.device):
    """Load the TripoSR model from Hugging Face (cached)."""
    device_key = str(device)
    if device_key in _model_cache:
        return _model_cache[device_key]

    from tsr.system import TSR

    logging.info("Loading TripoSR model from Hugging Face...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    _model_cache[device_key] = model
    logging.info("Model loaded successfully")
    return model


def preprocess_for_triposr(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    """Preprocess image for TripoSR: remove bg, resize foreground, gray bg."""
    import rembg
    from tsr.utils import remove_background, resize_foreground

    rembg_session = rembg.new_session()
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, foreground_ratio)

    # TripoSR expects RGB with 50% gray background
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def image_to_mesh(
    image_path: str,
    output_path: str,
    device: Optional[torch.device] = None,
    mc_resolution: int = 256,
    foreground_ratio: float = 0.85,
) -> str:
    """Convert a 2D image to a 3D mesh using TripoSR.

    Args:
        image_path: Path to input image
        output_path: Path for output mesh file (.obj or .glb)
        device: Torch device (auto-detected if None)
        mc_resolution: Marching cubes resolution (higher = more detail)
        foreground_ratio: Foreground size ratio for preprocessing

    Returns:
        Path to the output mesh file
    """
    if device is None:
        device = get_device()

    image = Image.open(image_path)
    processed = preprocess_for_triposr(image, foreground_ratio)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_triposr(device)

    with torch.no_grad():
        scene_codes = model([processed], device=device)

    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
    mesh = meshes[0]

    # Save mesh
    output = Path(output_path)
    if output.suffix == ".glb":
        mesh.export(str(output), file_type="glb")
    else:
        mesh.export(str(output))

    vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 'unknown'
    face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 'unknown'
    logging.info(f"Saved mesh to {output_path}")
    logging.info(f"  Vertices: {vertex_count}")
    logging.info(f"  Faces: {face_count}")

    return str(output)


def batch_image_to_mesh(
    items: list,
    device: Optional[torch.device] = None,
    mc_resolution: int = 256,
    foreground_ratio: float = 0.85,
) -> list:
    """Batch-process multiple images through TripoSR in one forward pass.

    Args:
        items: List of (image_path, output_path) tuples
        device: Torch device
        mc_resolution: Marching cubes resolution

    Returns:
        List of (output_path, success, error) tuples
    """
    if not items:
        return []
    if device is None:
        device = get_device()

    model = load_triposr(device)
    results = []

    # Preprocess all images
    processed_images = []
    valid_items = []
    for image_path, output_path in items:
        try:
            image = Image.open(image_path)
            processed = preprocess_for_triposr(image, foreground_ratio)
            processed_images.append(processed)
            valid_items.append((image_path, output_path))
        except Exception as e:
            results.append((output_path, False, str(e)))

    if not processed_images:
        return results

    # Batch inference
    with torch.no_grad():
        scene_codes = model(processed_images, device=device)

    # Extract meshes
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)

    for mesh, (image_path, output_path) in zip(meshes, valid_items):
        try:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".glb":
                mesh.export(str(output), file_type="glb")
            else:
                mesh.export(str(output))
            results.append((output_path, True, None))
        except Exception as e:
            results.append((output_path, False, str(e)))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image to 3D mesh using TripoSR")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, help="Output mesh path (.obj or .glb)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Marching cubes resolution (default: 256)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (skip GPU)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    out = args.output or f"{Path(args.input).stem}.obj"

    image_to_mesh(args.input, out, device=device, mc_resolution=args.resolution)

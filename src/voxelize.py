"""
Mesh-to-voxel conversion with surface color sampling.

Converts a triangle mesh into a voxel grid at configurable resolution.
Uses surface-based color sampling (not voxel-center) for accurate colors.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh")
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import trimesh


def voxelize_mesh(
    mesh_path: str,
    resolution: int = 32,
    output_path: Optional[str] = None,
    name: Optional[str] = None,
) -> dict:
    """Convert a triangle mesh to a colored voxel grid.

    Args:
        mesh_path: Path to input mesh (.obj, .glb, .ply, etc.)
        resolution: Voxel grid resolution (default 32)
        output_path: Optional path to save voxel data
        name: Asset name (emoji char or image name)

    Returns:
        Dict with voxel data
    """
    scene_or_mesh = trimesh.load(mesh_path, process=False)

    # Handle scene vs mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No valid meshes found in scene")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    # Normalize mesh to fit in unit cube centered at origin
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()
    if scale > 0:
        mesh.vertices = (mesh.vertices - center) / scale

    # Voxelize using trimesh
    pitch = 1.0 / resolution
    voxel_grid = mesh.voxelized(pitch)
    filled = voxel_grid.matrix

    # Remove interior voxels (keep only surface shell)
    from scipy import ndimage
    eroded = ndimage.binary_erosion(filled)
    surface = filled & ~eroded

    # Sample colors from the mesh surface
    colors = _sample_surface_colors(mesh, voxel_grid, surface, resolution)

    # Build voxel data
    voxels = []
    indices = np.argwhere(surface)
    for idx, (ix, iy, iz) in enumerate(indices):
        r, g, b = colors[idx]
        voxels.append({
            "x": int(ix), "y": int(iy), "z": int(iz),
            "r": int(r), "g": int(g), "b": int(b),
        })

    # Also build compact format: flat arrays for efficiency
    n = len(voxels)
    positions = []
    color_data = []
    for v in voxels:
        positions.extend([v["x"], v["y"], v["z"]])
        color_data.extend([v["r"], v["g"], v["b"]])

    result = {
        "name": name or Path(mesh_path).stem,
        "resolution": resolution,
        "count": n,
        "bounds": {
            "min": [int(indices[:, 0].min()), int(indices[:, 1].min()), int(indices[:, 2].min())] if n > 0 else [0, 0, 0],
            "max": [int(indices[:, 0].max()), int(indices[:, 1].max()), int(indices[:, 2].max())] if n > 0 else [0, 0, 0],
        },
        # Compact arrays: positions as [x0,y0,z0,x1,y1,z1,...], colors as [r0,g0,b0,...]
        "positions": positions,
        "colors": color_data,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f)

    return result


def _sample_surface_colors(
    mesh: trimesh.Trimesh,
    voxel_grid,
    surface_mask: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Sample colors from the nearest mesh surface for each surface voxel.

    Uses closest-point queries on the mesh surface rather than voxel-center
    sampling, which avoids the problem of centers landing inside empty space.
    """
    indices = np.argwhere(surface_mask)
    n = len(indices)

    # Convert voxel indices to world coordinates (centers of voxels)
    voxel_origins = voxel_grid.transform[:3, 3]
    voxel_scale = voxel_grid.scale
    if hasattr(voxel_grid, 'pitch'):
        pitch = voxel_grid.pitch
    else:
        pitch = voxel_scale
    world_points = indices * pitch + voxel_origins + pitch / 2

    # Find closest points on mesh surface
    closest_points, _, face_indices = mesh.nearest.on_surface(world_points)

    # Get colors at those surface points
    colors = np.full((n, 3), 180, dtype=np.uint8)  # default gray

    if mesh.visual is not None:
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            # Vertex colors: interpolate from face vertices
            try:
                vc = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
                for i, fi in enumerate(face_indices):
                    if fi < 0 or fi >= len(mesh.faces):
                        continue
                    face_verts = mesh.faces[fi]
                    face_colors = vc[face_verts]
                    colors[i] = face_colors.mean(axis=0).astype(np.uint8)
            except (IndexError, AttributeError):
                pass

        elif hasattr(mesh.visual, 'material') and hasattr(mesh.visual, 'uv'):
            # Texture: sample from material image
            try:
                material = mesh.visual.material
                if hasattr(material, 'image') and material.image is not None:
                    tex_img = np.array(material.image.convert("RGB"))
                    uvs = mesh.visual.uv
                    th, tw = tex_img.shape[:2]
                    for i, fi in enumerate(face_indices):
                        if fi < 0 or fi >= len(mesh.faces):
                            continue
                        face_verts = mesh.faces[fi]
                        face_uvs = uvs[face_verts]
                        avg_uv = face_uvs.mean(axis=0)
                        px = int(np.clip(avg_uv[0] * tw, 0, tw - 1))
                        py = int(np.clip((1 - avg_uv[1]) * th, 0, th - 1))
                        colors[i] = tex_img[py, px]
            except (IndexError, AttributeError):
                pass

    return colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize a 3D mesh")
    parser.add_argument("input", type=str, help="Path to input mesh")
    parser.add_argument("--resolution", "-r", type=int, default=32,
                        help="Voxel resolution (default: 32)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON path")
    parser.add_argument("--name", type=str, help="Asset name")
    args = parser.parse_args()

    out = args.output or f"{Path(args.input).stem}_voxels.json"
    voxelize_mesh(args.input, resolution=args.resolution, output_path=out, name=args.name)

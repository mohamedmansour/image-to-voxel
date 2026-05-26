"""
Export voxel data in compact format for web loading.

Supports individual JSON files per asset and a combined atlas.
Uses compact flat arrays instead of per-voxel objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def build_atlas(voxel_dir: str, output_path: str) -> None:
    """Combine individual voxel JSON files into a single atlas.

    The atlas uses compact arrays for efficiency:
    - positions: flat [x0,y0,z0, x1,y1,z1, ...] array
    - colors: flat [r0,g0,b0, r1,g1,b1, ...] array
    """
    voxel_path = Path(voxel_dir)
    atlas: Dict[str, dict] = {}

    files = sorted(voxel_path.glob("*.json"))
    if not files:
        print(f"No voxel JSON files found in {voxel_dir}")
        return

    for f in files:
        if f.name == "voxel_atlas.json":
            continue
        with open(f) as fp:
            data = json.load(fp)

        name = data.get("name", f.stem)
        atlas[name] = {
            "resolution": data["resolution"],
            "count": data["count"],
            "bounds": data["bounds"],
            "positions": data["positions"],
            "colors": data["colors"],
        }
        print(f"  Added: {name} ({data['count']} voxels)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(atlas, f)

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"\nAtlas saved to {output_path}")
    print(f"  Assets: {len(atlas)}")
    print(f"  Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build voxel atlas from individual files")
    parser.add_argument("voxel_dir", type=str, help="Directory containing voxel JSON files")
    parser.add_argument("--output", "-o", type=str, default="voxel_atlas.json",
                        help="Output atlas path")
    args = parser.parse_args()
    build_atlas(args.voxel_dir, args.output)

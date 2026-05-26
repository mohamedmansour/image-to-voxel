"""
Image preparation and stylization for the image-to-voxel pipeline.

Two input modes:
  - Emoji mode: renders an emoji character to an image
  - Image mode: loads any image file (PNG, JPG, etc.)

Then applies stylization (background removal, shading, AO) to help
the 3D reconstruction model produce better meshes.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage


def render_emoji(emoji: str, size: int = 512) -> Image.Image:
    """Render an emoji character to a PIL Image on a white background."""
    font = None

    if platform.system() == "Darwin":
        # Apple Color Emoji only supports specific sizes (multiples of 16 up to 160)
        # Render at the largest supported size then scale up
        for font_size in [160, 96, 64, 48, 32]:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Apple Color Emoji.ttc", font_size
                )
                break
            except (OSError, IOError):
                continue

    if font is None:
        for name in ["NotoColorEmoji.ttf", "Segoe UI Emoji", "Arial"]:
            try:
                font = ImageFont.truetype(name, int(size * 0.72))
                break
            except (OSError, IOError):
                continue

    if font is None:
        font = ImageFont.load_default()

    # Render at native font size first
    bbox = font.getbbox(emoji)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(tw, th) // 4
    canvas_w, canvas_h = tw + pad * 2, th + pad * 2

    img = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = pad - bbox[0]
    y = pad - bbox[1]
    draw.text((x, y), emoji, font=font, embedded_color=True)

    # Scale up to target size
    img = img.resize((size, size), Image.LANCZOS)

    return img


def load_image(path: str, size: int = 512) -> Image.Image:
    """Load any image file, resize/pad to square, center the subject."""
    img = Image.open(path).convert("RGBA")

    # Resize maintaining aspect ratio, pad to square
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center on white background
    result = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    result.paste(img, (offset_x, offset_y), img)

    return result


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using rembg, place subject on white."""
    from rembg import remove

    # rembg works on the raw image bytes
    result = remove(img)
    # Composite onto white background
    white = Image.new("RGBA", result.size, (255, 255, 255, 255))
    white.paste(result, mask=result.split()[3])
    return white


def apply_stylization(img: Image.Image) -> Image.Image:
    """Apply shading and depth cues to help 3D reconstruction."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # Convert to grayscale for edge/depth analysis
    gray = np.mean(arr[:, :, :3], axis=2)

    # Detect the subject mask (non-white pixels)
    white_thresh = 240
    subject_mask = np.any(arr[:, :, :3] < white_thresh, axis=2).astype(np.float32)

    if subject_mask.sum() < 100:
        # No significant subject found, return as-is
        return img

    # Find subject centroid
    coords = np.argwhere(subject_mask > 0)
    cy, cx = coords.mean(axis=0)

    # --- Edge-aware shading ---
    # Compute distance from subject edges (inward)
    dist_from_edge = ndimage.distance_transform_edt(subject_mask)
    max_dist = dist_from_edge.max()
    if max_dist > 0:
        dist_norm = dist_from_edge / max_dist
    else:
        dist_norm = dist_from_edge

    # --- Radial gradient for volume ---
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radial = radial[subject_mask > 0].max() if subject_mask.sum() > 0 else 1.0
    radial_norm = np.clip(radial / max(max_radial, 1.0), 0, 1)

    # Combine: lighter at center/top, darker at edges/bottom
    # Simulate top-left light source
    light_dir_x, light_dir_y = -0.3, -0.5
    light_map = 1.0 - 0.15 * (
        light_dir_x * (xx - cx) / max(max_radial, 1) +
        light_dir_y * (yy - cy) / max(max_radial, 1)
    )
    light_map = np.clip(light_map, 0.7, 1.15)

    # Edge darkening (ambient occlusion)
    ao = 0.85 + 0.15 * dist_norm
    ao = ndimage.gaussian_filter(ao, sigma=3)

    # Combine shading
    shading = light_map * ao
    shading = np.clip(shading, 0.6, 1.2)

    # Apply shading only to the subject
    for c in range(3):
        channel = arr[:, :, c]
        # Only shade non-white areas
        shaded = channel * shading
        arr[:, :, c] = np.where(subject_mask > 0, shaded, channel)

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # --- Soft drop shadow ---
    shadow_offset = int(h * 0.02)
    shadow_mask = ndimage.shift(subject_mask, [shadow_offset, shadow_offset // 2])
    shadow_mask = ndimage.gaussian_filter(shadow_mask, sigma=8) * 0.3
    for c in range(3):
        arr[:, :, c] = np.clip(
            arr[:, :, c].astype(np.float32) * (1 - shadow_mask * 0.5),
            0, 255
        ).astype(np.uint8)

    return Image.fromarray(arr)


def prepare_image(
    emoji: Optional[str] = None,
    image_path: Optional[str] = None,
    size: int = 512,
    skip_rembg: bool = False,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Main entry point: prepare an image for 3D reconstruction."""
    if emoji:
        img = render_emoji(emoji, size)
        # Emojis on white bg usually don't need rembg
        skip_rembg = True
    elif image_path:
        img = load_image(image_path, size)
    else:
        raise ValueError("Must provide either --emoji or --image")

    # Background removal for arbitrary images
    if not skip_rembg:
        img = remove_background(img)

    # Apply stylization
    img = apply_stylization(img)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)


    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare image for 3D reconstruction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emoji", type=str, help="Emoji character to render")
    group.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--size", type=int, default=512, help="Output size (default: 512)")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--skip-rembg", action="store_true", help="Skip background removal")
    args = parser.parse_args()

    out = args.output
    if not out:
        if args.emoji:
            out = f"prepared_{ord(args.emoji[0]):x}.png"
        else:
            out = f"prepared_{Path(args.image).stem}.png"

    prepare_image(
        emoji=args.emoji,
        image_path=args.image,
        size=args.size,
        skip_rembg=args.skip_rembg,
        output_path=out,
    )

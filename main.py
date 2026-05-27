#!/usr/bin/env python3
"""
Normal Map / Heightmap Hillshade Viewer
======================================

Interactive viewer for rendering low-relief heritage objects from RGB normal maps
and single-band height/depth maps.

Designed for:
- coins
- faded inscriptions
- seals
- tablets
- low-relief carved or stamped surfaces

Features:
- fast PyQtGraph preview
- zoom / pan
- hemisphere light control
- RGB normal-map rendering
- single-band height/depth-map rendering
- lazy rasterio-backed loading for very large TIFF heightmaps
- sampled statistics for large rasters
- tiled BigTIFF/GeoTIFF full-resolution export with halo overlap
- seam-safe tiled export with global render normalization
- preview/export rotation in 90° steps
- multiple normal/height enhancement modes
- PNG/TIFF metadata embedding
- default output name: {input_filename}_hillshaded.png
- interactive preview resolution control, defaulting to 50%

Dependencies:
    pip install numpy pillow pyqtgraph PyQt6
    pip install rasterio   # recommended for very large TIFF height/depth maps

Run:
    python normal_map_heightmap_hillshade_viewer_v1_1_2_tile_seam_fix.py
    python normal_map_heightmap_hillshade_viewer_v1_1_2_tile_seam_fix.py normal_map.png
    python normal_map_heightmap_hillshade_viewer_v1_1_2_tile_seam_fix.py heightmap.tif
"""

from pathlib import Path
import argparse
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL import TiffImagePlugin


try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
except Exception:
    rasterio = None
    Resampling = None
    Window = None

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


APP_NAME = "Normal Map / Heightmap Hillshade Viewer"
APP_VERSION = "1.1.4 preview-export contrast match - 20260527"
DEVELOPER_CREDIT = "Developed by Sjors Nab (Utrecht University ArtLab, s.h.nab@uu.nl)"
LOGO_FILENAME = "logo.png"

pg.setConfigOptions(imageAxisOrder="row-major")

try:
    Signal = QtCore.Signal
except AttributeError:
    Signal = QtCore.pyqtSignal


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def default_output_for_input(input_path: str | Path | None, suffix: str = "_hillshaded") -> str:
    """
    Return default output filename based on input filename.

    Example:
        coin_normal.png -> coin_normal_hillshaded.png
    """
    if not input_path:
        return "hillshade_output.png"

    path = Path(input_path)
    return str(path.with_name(f"{path.stem}{suffix}.png"))


def find_logo_path() -> Path | None:
    """
    Find logo.png next to the script file, falling back to the current working
    directory. This keeps deployment simple: place logo.png in the same folder
    as this Python file.
    """
    candidates = []

    try:
        candidates.append(Path(__file__).resolve().with_name(LOGO_FILENAME))
    except NameError:
        pass

    candidates.append(Path.cwd() / LOGO_FILENAME)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def safe_mode_name(text: str) -> str:
    """
    Convert a render-mode label to a filename-safe suffix.
    """
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "render"


def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def normalize_vectors(n: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(n, axis=2, keepdims=True)
    return n / np.maximum(length, 1e-8)


def normalize_image_float(img: np.ndarray, percentile_clip: bool = True) -> np.ndarray:
    """
    Normalize an arbitrary float image to [0, 1] for display/export.
    Uses robust percentile scaling by default.
    """
    img = np.asarray(img, dtype=np.float32)

    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)

    values = img[finite]

    if percentile_clip:
        lo, hi = np.percentile(values, [1.0, 99.0])
    else:
        lo, hi = float(values.min()), float(values.max())

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    out = (img - lo) / (hi - lo)
    return clamp01(out)


def to_uint8_gray(img01: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(img01 * 255.0, 0, 255).astype(np.uint8))


def invert_rendered_uint8(rendered: np.ndarray) -> np.ndarray:
    """
    Invert output tones for either grayscale or RGB uint8 renderings.
    Useful for checking faint low-relief features under reversed tonal contrast.
    """
    return np.ascontiguousarray(255 - rendered.astype(np.uint8))


def normalize_rotation_degrees(rotation_degrees: int | float) -> int:
    """
    Normalize preview/export rotation to one of 0, 90, 180, or 270 degrees clockwise.
    """
    return int(round(float(rotation_degrees) / 90.0) * 90) % 360


def rotate_rendered_array(rendered: np.ndarray, rotation_degrees_clockwise: int | float) -> np.ndarray:
    """
    Rotate a rendered grayscale or RGB image in 90-degree steps.

    The rotation is applied after rendering, so it changes the output image
    orientation without changing light direction or normal/depth interpretation.
    """
    rotation = normalize_rotation_degrees(rotation_degrees_clockwise)

    if rotation == 0:
        return np.ascontiguousarray(rendered)
    if rotation == 90:
        return np.ascontiguousarray(np.rot90(rendered, k=-1, axes=(0, 1)))
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(rendered, k=2, axes=(0, 1)))
    if rotation == 270:
        return np.ascontiguousarray(np.rot90(rendered, k=1, axes=(0, 1)))

    # normalize_rotation_degrees should make this unreachable, but keep a safe fallback.
    return np.ascontiguousarray(rendered)


def rotated_dimensions(width: int, height: int, rotation_degrees_clockwise: int | float):
    """
    Return output width/height after a 90-degree-step clockwise rotation.
    """
    rotation = normalize_rotation_degrees(rotation_degrees_clockwise)
    width = int(width)
    height = int(height)
    if rotation in {90, 270}:
        return height, width
    return width, height


def rotated_window_for_source_tile(
    x: int,
    y: int,
    width: int,
    height: int,
    source_width: int,
    source_height: int,
    rotation_degrees_clockwise: int | float,
):
    """
    Map an unrotated source tile to its destination window after whole-image rotation.

    Returns (dst_x, dst_y, dst_width, dst_height), matching rasterio Window order.
    """
    rotation = normalize_rotation_degrees(rotation_degrees_clockwise)
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)
    source_width = int(source_width)
    source_height = int(source_height)

    if rotation == 0:
        return x, y, width, height
    if rotation == 90:
        return source_height - y - height, x, height, width
    if rotation == 180:
        return source_width - x - width, source_height - y - height, width, height
    if rotation == 270:
        return y, source_width - x - width, height, width

    return x, y, width, height


def apply_gamma_and_alpha(img01: np.ndarray, alpha: np.ndarray, gamma: float) -> np.ndarray:
    img01 = clamp01(img01)

    if gamma != 1.0:
        img01 = img01 ** gamma

    # Composite transparent/background areas onto white
    img01 = img01 * alpha + (1.0 - alpha) * 1.0
    return clamp01(img01)


def pil_to_qpixmap_scaled(path: Path, width: int, height: int) -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(str(path))
    return pixmap.scaled(
        width,
        height,
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )


# ---------------------------------------------------------------------------
# Normal-map / heightmap loading and decoding
# ---------------------------------------------------------------------------

def decode_normal_map_from_pil(img: Image.Image):
    """
    Decode RGB normal map from [0,255] to normalized [-1,1].

    Assumes:
      R = X
      G = Y
      B = Z

    Returns:
      nx, ny, nz, alpha
    """
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.float32)

    rgb = arr[..., :3] / 255.0
    alpha = arr[..., 3] / 255.0

    n = rgb * 2.0 - 1.0
    n = normalize_vectors(n)

    n = np.ascontiguousarray(n, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)

    return n[..., 0], n[..., 1], n[..., 2], alpha

def is_probably_heightmap(img: Image.Image) -> bool:
    """
    Heuristic for auto mode: single-band and float/integer images are treated
    as height/depth maps; RGB/RGBA images are treated as normal maps.
    """
    if img.mode in {"F", "I", "I;16", "I;16B", "I;16L", "L", "LA", "P"}:
        return True
    try:
        arr = np.asarray(img)
        return arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1)
    except Exception:
        return False


def height_array_and_alpha_from_pil(img: Image.Image):
    """
    Read a PIL image as a float32 height/depth array plus alpha/validity mask.

    - Single-band float TIFFs, such as RealityScan DSM/depth exports, are kept
      as numeric float values.
    - RGB images selected as height maps are converted to luminance.
    - NaN/Inf pixels become transparent in the alpha mask and are filled before
      gradient computation.
    """
    arr = np.asarray(img)

    alpha_from_image = None

    if arr.ndim == 3:
        # Preserve alpha if present, then convert RGB to luminance.
        if arr.shape[2] >= 4:
            a = arr[..., 3].astype(np.float32)
            max_a = np.nanmax(a) if np.isfinite(a).any() else 255.0
            alpha_from_image = a / max(max_a, 1.0)
            rgb = arr[..., :3].astype(np.float32)
        else:
            rgb = arr[..., :3].astype(np.float32)

        # ITU-R BT.601 luma coefficients; works for grayscale RGB too.
        height = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        height = arr.astype(np.float32)

    height = np.asarray(height, dtype=np.float32)
    finite = np.isfinite(height)

    if alpha_from_image is None:
        alpha = finite.astype(np.float32)
    else:
        alpha = (np.asarray(alpha_from_image, dtype=np.float32) * finite.astype(np.float32))

    return np.ascontiguousarray(height), np.ascontiguousarray(alpha)


def robust_normalize_height(height: np.ndarray, alpha: np.ndarray, clip_percentiles=(1.0, 99.0)):
    """
    Convert an arbitrary height/depth raster to [0,1] for visual hillshading.
    Percentile clipping suppresses extreme spikes/no-data values without
    changing the source file.
    """
    height = np.asarray(height, dtype=np.float32)
    valid = np.isfinite(height) & (np.asarray(alpha) > 0)

    if not valid.any():
        return np.zeros_like(height, dtype=np.float32), {"min": None, "max": None, "p1": None, "p99": None}

    values = height[valid].astype(np.float64)
    lo, hi = np.percentile(values, clip_percentiles)
    true_min = float(np.nanmin(values))
    true_max = float(np.nanmax(values))

    if hi <= lo:
        hi = float(np.nanmax(values))
        lo = float(np.nanmin(values))

    if hi <= lo:
        normalized = np.zeros_like(height, dtype=np.float32)
    else:
        filled = np.where(valid, height, lo)
        clipped = np.clip(filled, lo, hi)
        normalized = (clipped - lo) / (hi - lo)

    stats = {"min": true_min, "max": true_max, "p1": float(lo), "p99": float(hi)}
    return np.ascontiguousarray(normalized.astype(np.float32)), stats



def effective_radius_for_spacing(radius: int, pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0) -> int:
    """
    Convert a radius expressed in full-resolution/source pixels to the current
    working array resolution.

    This keeps preview and full-resolution export visually consistent when the
    preview is downsampled. For example, a 10% preview has a spacing of about
    10 source pixels per preview pixel, so a 12 px source-radius becomes about
    1 preview pixel.
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return 0

    try:
        avg_spacing = 0.5 * (abs(float(pixel_spacing_x)) + abs(float(pixel_spacing_y)))
    except Exception:
        avg_spacing = 1.0

    if not np.isfinite(avg_spacing) or avg_spacing <= 1e-8:
        avg_spacing = 1.0

    return max(1, int(round(radius / avg_spacing)))


def safe_gradient_2d(img: np.ndarray, pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0):
    """
    np.gradient wrapper using source-pixel spacing.

    The first returned derivative is along rows/Y; the second is along
    columns/X. Using spacing is essential for downsampled previews: without it,
    height slopes are exaggerated by roughly the preview downsampling factor,
    making the preview and full-resolution export disagree.
    """
    sx = float(pixel_spacing_x) if np.isfinite(float(pixel_spacing_x)) and float(pixel_spacing_x) > 0 else 1.0
    sy = float(pixel_spacing_y) if np.isfinite(float(pixel_spacing_y)) and float(pixel_spacing_y) > 0 else 1.0
    return np.gradient(np.asarray(img, dtype=np.float32), sy, sx)

def heightmap_to_normal_channels(
    img: Image.Image,
    strength: float = 50.0,
    invert_height: bool = False,
    smooth_radius: int = 0,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
):
    """
    Convert a height/depth map to normal channels compatible with the existing
    renderer.

    Coordinate system matches the original viewer:
      X = right, Y = down, Z = out of image.

    The height values are robust-normalized first because heritage relief maps
    often mix metric float values, NoData pixels, and isolated spikes. The
    strength parameter is therefore a visual relief exaggeration control rather
    than a calibrated metric z-scale.

    pixel_spacing_x/y describe how many full-resolution/source pixels one
    working-array pixel represents. Downsampled previews therefore use larger
    spacing values, preventing preview gradients from being exaggerated.
    """
    height, alpha = height_array_and_alpha_from_pil(img)
    h01, stats = robust_normalize_height(height, alpha)

    if invert_height:
        h01 = 1.0 - h01

    effective_smooth_radius = effective_radius_for_spacing(
        smooth_radius,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
    )
    if effective_smooth_radius > 0:
        h01 = box_blur_2d(h01, effective_smooth_radius)

    # np.gradient returns d/drow and d/dcolumn. Row is image Y/down;
    # column is image X/right. Spacing is in source pixels, so previews
    # and full-resolution exports use comparable slopes.
    dz_dy, dz_dx = safe_gradient_2d(
        h01.astype(np.float32),
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
    )

    strength = float(max(0.0, strength))
    nx = -dz_dx * strength
    ny = -dz_dy * strength
    nz = np.ones_like(h01, dtype=np.float32)

    n = np.stack([nx, ny, nz], axis=2)
    n = normalize_vectors(n)

    return (
        np.ascontiguousarray(n[..., 0], dtype=np.float32),
        np.ascontiguousarray(n[..., 1], dtype=np.float32),
        np.ascontiguousarray(n[..., 2], dtype=np.float32),
        np.ascontiguousarray(alpha, dtype=np.float32),
        stats,
    )



# ---------------------------------------------------------------------------
# Large raster height/depth-map support
# ---------------------------------------------------------------------------

LARGE_RASTER_PREVIEW_MAX_PIXELS = 6_000_000
LARGE_RASTER_STATS_MAX_PIXELS = 2_000_000
LARGE_RASTER_TILE_SIZE = 1024


def height_array_to_normal_channels(
    height: np.ndarray,
    alpha: np.ndarray | None = None,
    stats: dict | None = None,
    strength: float = 50.0,
    invert_height: bool = False,
    smooth_radius: int = 0,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
):
    """
    Convert an already-loaded height/depth array to normal channels.

    This is the array-based version used by the large-raster code path. It
    avoids PIL conversion and can be called on preview arrays or individual
    export tiles.

    pixel_spacing_x/y describe how many full-resolution/source pixels one
    working-array pixel represents. Downsampled previews therefore use larger
    spacing values, preventing preview gradients from being exaggerated.
    """
    height = np.asarray(height, dtype=np.float32)

    if alpha is None:
        alpha = np.isfinite(height).astype(np.float32)
    else:
        alpha = np.asarray(alpha, dtype=np.float32) * np.isfinite(height).astype(np.float32)

    valid = np.isfinite(height) & (alpha > 0)

    if stats is not None and stats.get("p99") is not None and stats.get("p1") is not None:
        lo = float(stats["p1"])
        hi = float(stats["p99"])
        if hi <= lo:
            hi = lo + 1.0
        filled = np.where(valid, height, lo)
        clipped = np.clip(filled, lo, hi)
        h01 = (clipped - lo) / (hi - lo)
        used_stats = stats
    else:
        h01, used_stats = robust_normalize_height(height, alpha)

    h01 = np.asarray(h01, dtype=np.float32)

    if invert_height:
        h01 = 1.0 - h01

    effective_smooth_radius = effective_radius_for_spacing(
        smooth_radius,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
    )
    if effective_smooth_radius > 0:
        h01 = box_blur_2d(h01, effective_smooth_radius)

    dz_dy, dz_dx = safe_gradient_2d(
        h01.astype(np.float32),
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
    )

    strength = float(max(0.0, strength))
    nx = -dz_dx * strength
    ny = -dz_dy * strength
    nz = np.ones_like(h01, dtype=np.float32)

    n = np.stack([nx, ny, nz], axis=2)
    n = normalize_vectors(n)

    return (
        np.ascontiguousarray(n[..., 0], dtype=np.float32),
        np.ascontiguousarray(n[..., 1], dtype=np.float32),
        np.ascontiguousarray(n[..., 2], dtype=np.float32),
        np.ascontiguousarray(alpha, dtype=np.float32),
        used_stats,
    )



class LargeHeightMapSource:
    """
    Rasterio-backed lazy source for very large single-band height/depth TIFFs.

    The source never loads the full image into RAM. It reads downsampled previews
    for interaction and reads haloed windows for full-resolution export.
    """
    def __init__(self, path: str | Path):
        if rasterio is None:
            raise RuntimeError(
                "rasterio is required for very large tiled height/depth maps. "
                "Install it with: pip install rasterio"
            )

        self.path = Path(path)
        self.dataset = rasterio.open(self.path)
        self.width = int(self.dataset.width)
        self.height = int(self.dataset.height)
        self.count = int(self.dataset.count)
        self.dtypes = tuple(self.dataset.dtypes)
        self.nodata = self.dataset.nodata
        self.crs = self.dataset.crs
        self.transform = self.dataset.transform
        self.stats = None

        if self.count < 1:
            self.close()
            raise ValueError("Raster has no bands.")

    def close(self):
        try:
            self.dataset.close()
        except Exception:
            pass

    @property
    def size(self):
        return (self.width, self.height)

    def is_single_band_heightmap(self) -> bool:
        return self.count == 1

    def _alpha_from_masked(self, arr):
        mask = getattr(arr, "mask", None)
        data = np.asarray(arr.filled(np.nan) if hasattr(arr, "filled") else arr, dtype=np.float32)

        if mask is None or mask is np.ma.nomask:
            valid = np.isfinite(data)
        else:
            valid = ~np.asarray(mask)
            if valid.shape == ():
                valid = np.ones(data.shape, dtype=bool) if bool(valid) else np.zeros(data.shape, dtype=bool)
            valid = valid & np.isfinite(data)

        if self.nodata is not None:
            try:
                valid = valid & (data != float(self.nodata))
            except Exception:
                pass

        alpha = valid.astype(np.float32)
        data = np.where(valid, data, np.nan).astype(np.float32)
        return data, alpha

    def read_resampled(self, out_width: int, out_height: int):
        out_width = max(1, int(out_width))
        out_height = max(1, int(out_height))
        arr = self.dataset.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
            masked=True,
        )
        return self._alpha_from_masked(arr)

    def read_preview(self, percent: int, max_pixels: int = LARGE_RASTER_PREVIEW_MAX_PIXELS):
        percent = max(1, min(100, int(percent)))
        out_width = max(1, int(round(self.width * percent / 100.0)))
        out_height = max(1, int(round(self.height * percent / 100.0)))

        pixels = out_width * out_height
        if pixels > max_pixels:
            scale = math.sqrt(max_pixels / float(pixels))
            out_width = max(1, int(out_width * scale))
            out_height = max(1, int(out_height * scale))

        return self.read_resampled(out_width, out_height)

    def estimate_stats(self, max_pixels: int = LARGE_RASTER_STATS_MAX_PIXELS):
        if self.stats is not None:
            return self.stats

        scale = math.sqrt(max_pixels / float(max(1, self.width * self.height)))
        if scale >= 1.0:
            out_width, out_height = self.width, self.height
        else:
            out_width = max(1, int(self.width * scale))
            out_height = max(1, int(self.height * scale))

        height, alpha = self.read_resampled(out_width, out_height)
        _, stats = robust_normalize_height(height, alpha)
        self.stats = stats
        return stats

    def read_window_with_halo(self, x: int, y: int, width: int, height: int, halo: int):
        halo = max(0, int(halo))
        x0 = max(0, int(x) - halo)
        y0 = max(0, int(y) - halo)
        x1 = min(self.width, int(x) + int(width) + halo)
        y1 = min(self.height, int(y) + int(height) + halo)

        read_width = x1 - x0
        read_height = y1 - y0
        window = Window(x0, y0, read_width, read_height)
        arr = self.dataset.read(1, window=window, masked=True)
        data, alpha = self._alpha_from_masked(arr)

        crop = (
            int(y) - y0,
            int(y) - y0 + int(height),
            int(x) - x0,
            int(x) - x0 + int(width),
        )
        return data, alpha, crop


def try_open_large_heightmap_source(path: str | Path, forced_mode: str = "Auto detect"):
    """
    Return a LargeHeightMapSource for single-band rasters when appropriate.
    RGB normal maps continue to use the classic PIL path.
    """
    if rasterio is None:
        return None

    forced_mode = forced_mode or "Auto detect"
    if forced_mode == "RGB normal map":
        return None

    suffix = Path(path).suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        return None

    try:
        source = LargeHeightMapSource(path)
    except Exception:
        return None

    if forced_mode == "Height/depth map" and source.count >= 1:
        return source

    if source.is_single_band_heightmap():
        return source

    source.close()
    return None


def halo_for_large_height_render(mode: str, height_smooth_radius: int, local_radius: int) -> int:
    """
    Determine the overlap needed for tile rendering so gradients/blur do not
    show seams at tile borders.

    The halo must cover every operation that reaches outside the requested
    output tile: optional height smoothing, normal gradients, and any render-mode
    specific blur/gradient. Using additive margins is safer than simply taking
    the maximum when several operations are chained.
    """
    height_smooth_radius = int(max(0, height_smooth_radius))
    local_radius = int(max(0, local_radius))

    # Height smoothing + first derivative to build normals.
    halo = height_smooth_radius + 6

    # Local normal deviation blurs the normal field after the height-to-normal step.
    if mode == "Local normal deviation":
        halo += local_radius + 4

    # Curvature and normal-gradient modes take another spatial derivative.
    if mode in {"Curvature from normals", "Normal gradient magnitude"}:
        halo += 8

    # Keep halos bounded for performance while still large enough for typical UI values.
    return int(min(max(halo, 8), 512))


def render_height_tile_from_array(height, alpha, stats, params, light_override=None, mode_override=None):
    nx, ny, nz, tile_alpha, _ = height_array_to_normal_channels(
        height,
        alpha,
        stats=stats,
        strength=float(params.get("height_strength", 50.0)),
        invert_height=bool(params.get("invert_height", False)),
        smooth_radius=int(params.get("height_smooth_radius", 0)),
        pixel_spacing_x=float(params.get("pixel_spacing_x", 1.0)),
        pixel_spacing_y=float(params.get("pixel_spacing_y", 1.0)),
    )

    light = light_override if light_override is not None else params["light"]
    mode = mode_override if mode_override is not None else params["mode"]

    rendered = render_mode_image(
        nx,
        ny,
        nz,
        tile_alpha,
        mode=mode,
        light=light,
        ambient=params["ambient"],
        gamma=params["gamma"],
        flip_x=params["flip_x"],
        flip_y=params["flip_y"],
        flip_z=params["flip_z"],
        multi_count=params["multi_count"],
        local_radius=params["local_radius"],
        pixel_spacing_x=float(params.get("pixel_spacing_x", 1.0)),
        pixel_spacing_y=float(params.get("pixel_spacing_y", 1.0)),
    )

    if params.get("invert_tones", False):
        rendered = invert_rendered_uint8(rendered)

    return rendered


# ---------------------------------------------------------------------------
# Seam-safe tiled rendering helpers
# ---------------------------------------------------------------------------

DIRECT_01_RENDER_MODES = {
    # Single light is already a physical [0,1] Lambertian render. Keeping it
    # direct preserves the user's chosen ambient/gamma/light relationship.
    "Single light",
}

POSITIVE_GLOBAL_NORMALIZED_RENDER_MODES = {
    # Mean/max/min multi-light need global contrast normalization for large
    # heightmaps. The interactive preview historically stretched these modes
    # over the visible image, while v1.1.2/v1.1.3 exported direct [0,1] values.
    # That made exports look much darker than the preview, especially Min
    # multi-light. Global/sample-derived normalization keeps output seam-free
    # while matching the preview much more closely.
    "Mean multi-light",
    "Max multi-light",
    "Min multi-light",
    "Range multi-light",
    "Std-dev multi-light",
    "Slope from normals",
    "Local normal deviation",
    "Normal gradient magnitude",
}

SIGNED_GLOBAL_SCALE_RENDER_MODES = {
    "Curvature from normals",
}

RGB_DIRECT_RENDER_MODES = {
    "RGB 3-light composite",
}


def render_normalization_kind(mode: str) -> str:
    """
    Return the normalization strategy used by the seam-safe large-raster export.

    The original interactive renderer normalizes some enhancement modes per image.
    In a tiled export, doing that per tile causes visible seams because each tile
    gets its own contrast stretch. These strategies make the contrast consistent
    over the whole output.
    """
    if mode in RGB_DIRECT_RENDER_MODES:
        return "rgb_direct_01"
    if mode in DIRECT_01_RENDER_MODES:
        return "direct_01"
    if mode in SIGNED_GLOBAL_SCALE_RENDER_MODES:
        return "signed_global_scale"
    return "positive_global_percentile"


def compute_normal_render_raw(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    mode: str,
    light: np.ndarray,
    ambient: float,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    multi_count: int,
    local_radius: int,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
) -> np.ndarray:
    """
    Compute a floating-point render quantity before final contrast/gamma/alpha.

    This mirrors the original render modes but deliberately avoids per-tile
    percentile normalization. Final normalization is applied later using either
    fixed [0,1] ranges or global/sample-derived statistics.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)

    if mode == "Single light":
        return lambert_shade(nx, ny, nz, light, ambient)

    if mode in {
        "Mean multi-light",
        "Max multi-light",
        "Min multi-light",
        "Range multi-light",
        "Std-dev multi-light",
    }:
        _, altitude = light_vector_to_az_alt(light[0], light[1], light[2])
        altitude = max(1.0, min(89.0, altitude))
        lights = circular_light_vectors(multi_count, altitude)
        stack = compute_multi_light_stack(nx, ny, nz, lights, ambient)

        if mode == "Mean multi-light":
            return stack.mean(axis=2)
        if mode == "Max multi-light":
            return stack.max(axis=2)
        if mode == "Min multi-light":
            return stack.min(axis=2)
        if mode == "Range multi-light":
            return stack.max(axis=2) - stack.min(axis=2)
        if mode == "Std-dev multi-light":
            return stack.std(axis=2)

    if mode == "RGB 3-light composite":
        az, alt = light_vector_to_az_alt(light[0], light[1], light[2])
        lights = [
            az_alt_to_light_vector(az, alt),
            az_alt_to_light_vector((az + 120.0) % 360.0, alt),
            az_alt_to_light_vector((az + 240.0) % 360.0, alt),
        ]
        channels = [lambert_shade(nx, ny, nz, l, ambient) for l in lights]
        return np.stack(channels, axis=2).astype(np.float32)

    if mode == "Slope from normals":
        return np.sqrt(nx * nx + ny * ny).astype(np.float32)

    if mode == "Local normal deviation":
        effective_radius = effective_radius_for_spacing(
            local_radius,
            pixel_spacing_x=pixel_spacing_x,
            pixel_spacing_y=pixel_spacing_y,
        )
        bx, by, bz = box_blur_normals(nx, ny, nz, effective_radius)
        dot = np.clip(nx * bx + ny * by + nz * bz, -1.0, 1.0)
        return np.arccos(dot).astype(np.float32)

    if mode == "Curvature from normals":
        dnx_dy, dnx_dx = safe_gradient_2d(nx, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
        dny_dy, dny_dx = safe_gradient_2d(ny, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
        return (dnx_dx + dny_dy).astype(np.float32)

    if mode == "Normal gradient magnitude":
        nx_y, nx_x = safe_gradient_2d(nx, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
        ny_y, ny_x = safe_gradient_2d(ny, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
        nz_y, nz_x = safe_gradient_2d(nz, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
        return np.sqrt(
            nx_x * nx_x + nx_y * nx_y
            + ny_x * ny_x + ny_y * ny_y
            + nz_x * nz_x + nz_y * nz_y
        ).astype(np.float32)

    raise ValueError(f"Unknown render mode: {mode}")


def crop_raw_and_alpha(raw: np.ndarray, alpha: np.ndarray, crop):
    if crop is None:
        return raw, alpha

    y0, y1, x0, x1 = crop
    if raw.ndim == 3:
        raw = raw[y0:y1, x0:x1, :]
    else:
        raw = raw[y0:y1, x0:x1]
    alpha = alpha[y0:y1, x0:x1]
    return raw, alpha


def build_render_normalization_from_values(mode: str, values: np.ndarray) -> dict:
    kind = render_normalization_kind(mode)

    if kind in {"direct_01", "rgb_direct_01"}:
        return {"kind": kind}

    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        if kind == "signed_global_scale":
            return {"kind": kind, "scale": 1.0}
        return {"kind": kind, "lo": 0.0, "hi": 1.0}

    if kind == "signed_global_scale":
        scale = float(np.percentile(np.abs(values), 99.0))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        return {"kind": kind, "scale": scale}

    lo, hi = np.percentile(values, [1.0, 99.0])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values)) if values.size else 0.0
        hi = float(np.nanmax(values)) if values.size else 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0

    return {"kind": kind, "lo": lo, "hi": hi}


def estimate_large_render_normalization(source, height_stats: dict, params: dict, mode: str, light: np.ndarray) -> dict:
    """
    Estimate global contrast statistics for seam-free tiled export.

    Uses full-resolution sample windows distributed over the raster rather than
    a heavily downsampled preview. That keeps contrast statistics closer to the
    final export while still avoiding a full-raster pass.
    """
    kind = render_normalization_kind(mode)
    if kind in {"direct_01", "rgb_direct_01"}:
        return {"kind": kind}

    sample_size = int(min(768, max(128, LARGE_RASTER_TILE_SIZE)))
    sample_w = min(sample_size, int(source.width))
    sample_h = min(sample_size, int(source.height))

    if sample_w <= 0 or sample_h <= 0:
        return build_render_normalization_from_values(mode, np.array([], dtype=np.float32))

    if source.width <= sample_w:
        xs = [0]
    else:
        xs = [int(round(v)) for v in np.linspace(0, source.width - sample_w, 3)]

    if source.height <= sample_h:
        ys = [0]
    else:
        ys = [int(round(v)) for v in np.linspace(0, source.height - sample_h, 3)]

    halo = halo_for_large_height_render(
        mode,
        int(params.get("height_smooth_radius", 0)),
        int(params.get("local_radius", 0)),
    )

    collected = []
    max_values = LARGE_RASTER_STATS_MAX_PIXELS

    for y in ys:
        for x in xs:
            try:
                height, alpha, crop = source.read_window_with_halo(x, y, sample_w, sample_h, halo)
                nx, ny, nz, tile_alpha, _ = height_array_to_normal_channels(
                    height,
                    alpha,
                    stats=height_stats,
                    strength=float(params.get("height_strength", 50.0)),
                    invert_height=bool(params.get("invert_height", False)),
                    smooth_radius=int(params.get("height_smooth_radius", 0)),
                    pixel_spacing_x=float(params.get("pixel_spacing_x", 1.0)),
                    pixel_spacing_y=float(params.get("pixel_spacing_y", 1.0)),
                )

                raw = compute_normal_render_raw(
                    nx,
                    ny,
                    nz,
                    mode=mode,
                    light=light,
                    ambient=params["ambient"],
                    flip_x=params["flip_x"],
                    flip_y=params["flip_y"],
                    flip_z=params["flip_z"],
                    multi_count=params["multi_count"],
                    local_radius=params["local_radius"],
                    pixel_spacing_x=float(params.get("pixel_spacing_x", 1.0)),
                    pixel_spacing_y=float(params.get("pixel_spacing_y", 1.0)),
                )
                raw, tile_alpha = crop_raw_and_alpha(raw, tile_alpha, crop)
                valid = np.isfinite(raw) & (tile_alpha > 0)
                values = raw[valid]
                if values.size:
                    if values.size > max_values // 9:
                        step = max(1, int(math.ceil(values.size / float(max_values // 9))))
                        values = values[::step]
                    collected.append(np.asarray(values, dtype=np.float32))
            except Exception:
                # One bad sample should not block export; the fallback below will
                # still provide a stable normalization.
                continue

    if not collected:
        return build_render_normalization_from_values(mode, np.array([], dtype=np.float32))

    values = np.concatenate(collected)
    if values.size > max_values:
        step = max(1, int(math.ceil(values.size / float(max_values))))
        values = values[::step]

    return build_render_normalization_from_values(mode, values)


def finalize_raw_render_to_uint8(raw: np.ndarray, alpha: np.ndarray, gamma: float, render_norm: dict) -> np.ndarray:
    kind = (render_norm or {}).get("kind", "direct_01")

    if kind == "rgb_direct_01":
        out = clamp01(raw.astype(np.float32))
        if gamma != 1.0:
            out = out ** gamma
        out = out * alpha[..., None] + (1.0 - alpha[..., None]) * 1.0
        return np.ascontiguousarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))

    raw = np.asarray(raw, dtype=np.float32)

    if kind == "direct_01":
        out = clamp01(raw)
    elif kind == "signed_global_scale":
        scale = float((render_norm or {}).get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        out = 0.5 + 0.5 * np.clip(raw / scale, -1.0, 1.0)
    else:
        lo = float((render_norm or {}).get("lo", 0.0))
        hi = float((render_norm or {}).get("hi", 1.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        out = clamp01((raw - lo) / (hi - lo))

    out = apply_gamma_and_alpha(out, alpha, gamma)
    return to_uint8_gray(out)


def render_height_tile_from_array_seam_safe(
    height,
    alpha,
    stats,
    params,
    light_override=None,
    mode_override=None,
    crop=None,
    render_norm=None,
):
    """
    Render a large-raster tile using global/sample-derived normalization.

    Unlike render_height_tile_from_array(), this function is safe for tiled
    full-resolution export because all contrast stretching is global, not per
    tile. The optional crop is applied to the raw metric before final scaling so
    halo pixels influence gradients/blur but are not written to the output.
    """
    nx, ny, nz, tile_alpha, _ = height_array_to_normal_channels(
        height,
        alpha,
        stats=stats,
        strength=float(params.get("height_strength", 50.0)),
        invert_height=bool(params.get("invert_height", False)),
        smooth_radius=int(params.get("height_smooth_radius", 0)),
        pixel_spacing_x=float(params.get("pixel_spacing_x", 1.0)),
        pixel_spacing_y=float(params.get("pixel_spacing_y", 1.0)),
    )

    light = light_override if light_override is not None else params["light"]
    mode = mode_override if mode_override is not None else params["mode"]

    raw = compute_normal_render_raw(
        nx,
        ny,
        nz,
        mode=mode,
        light=light,
        ambient=params["ambient"],
        flip_x=params["flip_x"],
        flip_y=params["flip_y"],
        flip_z=params["flip_z"],
        multi_count=params["multi_count"],
        local_radius=params["local_radius"],
        pixel_spacing_x=float(params.get("pixel_spacing_x", 1.0)),
        pixel_spacing_y=float(params.get("pixel_spacing_y", 1.0)),
    )

    raw, tile_alpha = crop_raw_and_alpha(raw, tile_alpha, crop)

    if render_norm is None:
        render_norm = build_render_normalization_from_values(mode, raw[np.isfinite(raw)])

    rendered = finalize_raw_render_to_uint8(
        raw,
        tile_alpha,
        gamma=params["gamma"],
        render_norm=render_norm,
    )

    if params.get("invert_tones", False):
        rendered = invert_rendered_uint8(rendered)

    return rendered

def make_preview_image(img: Image.Image, max_preview_size: int) -> Image.Image:
    """
    Resize image for faster interactive preview.
    Full-resolution data is still used when saving.

    This is the legacy maximum-size limiter. The UI also provides a preview
    percentage control; both limits are respected.
    """
    if max_preview_size <= 0:
        return img.copy()

    preview = img.copy()
    preview.thumbnail((max_preview_size, max_preview_size), Image.Resampling.LANCZOS)
    return preview


def make_preview_image_percent(img: Image.Image, max_preview_size: int, percent: int) -> Image.Image:
    """
    Build the interactive preview image.

    percent is capped by the UI from 1 to 100 and is interpreted literally:
    100% means the full-resolution source image is used for the preview.

    max_preview_size is kept as a legacy command-line argument but is no longer
    applied here, because a hidden size cap makes the percentage control
    misleading. Use a lower preview percentage for performance.
    """
    percent = max(1, min(100, int(percent)))

    if percent >= 100:
        return img.copy()

    w, h = img.size
    new_w = max(1, int(round(w * percent / 100.0)))
    new_h = max(1, int(round(h * percent / 100.0)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Light geometry
# ---------------------------------------------------------------------------

def az_alt_to_light_vector(azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    """
    Azimuth convention:
      0°   = light from top / north
      90°  = light from right / east
      180° = light from bottom / south
      270° = light from left / west

    Coordinate system:
      X = right
      Y = down
      Z = out of image
    """
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)

    lx = np.sin(az) * np.cos(alt)
    ly = -np.cos(az) * np.cos(alt)
    lz = np.sin(alt)

    light = np.array([lx, ly, lz], dtype=np.float32)
    light /= np.linalg.norm(light)
    return light


def light_vector_to_az_alt(lx: float, ly: float, lz: float):
    """
    Convert image-coordinate light vector back to azimuth/altitude.
    """
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, float(lz)))))
    azimuth = math.degrees(math.atan2(float(lx), -float(ly))) % 360.0
    return azimuth, altitude


def circular_light_vectors(count: int, altitude_deg: float) -> np.ndarray:
    """
    Generate evenly spaced lights around the object at a fixed altitude.
    """
    count = max(3, int(count))
    return np.array(
        [az_alt_to_light_vector(i * 360.0 / count, altitude_deg) for i in range(count)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Fast local box blur, no scipy required
# ---------------------------------------------------------------------------

def box_blur_2d(img: np.ndarray, radius: int) -> np.ndarray:
    """
    Fast box blur using integral image.

    radius = 0 returns the input.
    Edges are padded with reflection.
    """
    radius = int(radius)
    if radius <= 0:
        return img.astype(np.float32, copy=False)

    img = np.asarray(img, dtype=np.float32)
    pad = radius
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")

    # Integral image with zero border
    integral = np.pad(
        padded.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )

    k = 2 * radius + 1

    out = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    ) / float(k * k)

    return np.ascontiguousarray(out.astype(np.float32))


def box_blur_normals(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, radius: int):
    bx = box_blur_2d(nx, radius)
    by = box_blur_2d(ny, radius)
    bz = box_blur_2d(nz, radius)

    n = np.stack([bx, by, bz], axis=2)
    n = normalize_vectors(n)

    return n[..., 0], n[..., 1], n[..., 2]


# ---------------------------------------------------------------------------
# Rendering algorithms
# ---------------------------------------------------------------------------

RENDER_MODES = [
    "Single light",
    "Mean multi-light",
    "Max multi-light",
    "Min multi-light",
    "Range multi-light",
    "Std-dev multi-light",
    "RGB 3-light composite",
    "Slope from normals",
    "Local normal deviation",
    "Curvature from normals",
    "Normal gradient magnitude",
]


def corrected_channels(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
):
    sx = -1.0 if flip_x else 1.0
    sy = -1.0 if flip_y else 1.0
    sz = -1.0 if flip_z else 1.0

    return sx * nx, sy * ny, sz * nz


def lambert_shade(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    light: np.ndarray,
    ambient: float,
) -> np.ndarray:
    """
    Lambertian normal-map shading for one light vector.
    Returns float image in [0,1].
    """
    lx, ly, lz = light
    shade = nx * lx + ny * ly + nz * lz
    shade = np.clip(shade, 0.0, 1.0)

    if ambient > 0:
        shade = ambient + (1.0 - ambient) * shade

    return clamp01(shade.astype(np.float32))


def render_single_light(
    nx, ny, nz, alpha, light, ambient, gamma, flip_x, flip_y, flip_z
) -> np.ndarray:
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)
    shade = lambert_shade(nx, ny, nz, light, ambient)
    shade = apply_gamma_and_alpha(shade, alpha, gamma)
    return to_uint8_gray(shade)


def compute_multi_light_stack(nx, ny, nz, lights, ambient: float):
    """
    Return stack of hillshades for many lights.
    Shape: H x W x N
    """
    shades = []
    for light in lights:
        shades.append(lambert_shade(nx, ny, nz, light, ambient))
    return np.stack(shades, axis=2).astype(np.float32)


def render_multi_light(
    nx,
    ny,
    nz,
    alpha,
    mode: str,
    light: np.ndarray,
    ambient: float,
    gamma: float,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    multi_count: int,
) -> np.ndarray:
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)

    _, altitude = light_vector_to_az_alt(light[0], light[1], light[2])
    altitude = max(1.0, min(89.0, altitude))

    lights = circular_light_vectors(multi_count, altitude)
    stack = compute_multi_light_stack(nx, ny, nz, lights, ambient)

    if mode == "Mean multi-light":
        out = stack.mean(axis=2)
        out = normalize_image_float(out, percentile_clip=False)
    elif mode == "Max multi-light":
        out = stack.max(axis=2)
        out = normalize_image_float(out, percentile_clip=False)
    elif mode == "Min multi-light":
        out = stack.min(axis=2)
        out = normalize_image_float(out, percentile_clip=False)
    elif mode == "Range multi-light":
        out = stack.max(axis=2) - stack.min(axis=2)
        out = normalize_image_float(out, percentile_clip=True)
    elif mode == "Std-dev multi-light":
        out = stack.std(axis=2)
        out = normalize_image_float(out, percentile_clip=True)
    else:
        raise ValueError(f"Unknown multi-light mode: {mode}")

    out = apply_gamma_and_alpha(out, alpha, gamma)
    return to_uint8_gray(out)


def render_rgb_three_light_composite(
    nx,
    ny,
    nz,
    alpha,
    light,
    ambient,
    gamma,
    flip_x,
    flip_y,
    flip_z,
) -> np.ndarray:
    """
    Orientation-colour composite from three lights 120 degrees apart.

    The current hemisphere altitude is reused. The current azimuth becomes the
    red-channel light. Green and blue are +120° and +240°.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)

    az, alt = light_vector_to_az_alt(light[0], light[1], light[2])
    lights = [
        az_alt_to_light_vector(az, alt),
        az_alt_to_light_vector((az + 120.0) % 360.0, alt),
        az_alt_to_light_vector((az + 240.0) % 360.0, alt),
    ]

    channels = []
    for l in lights:
        s = lambert_shade(nx, ny, nz, l, ambient)
        s = apply_gamma_and_alpha(s, alpha, gamma)
        channels.append(to_uint8_gray(s))

    rgb = np.stack(channels, axis=2)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def render_slope_from_normals(
    nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z
) -> np.ndarray:
    """
    Direction-independent relief strength.

    Equivalent to sin(surface slope), because sqrt(nx^2 + ny^2) is large where
    the surface normal is tilted away from the camera.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)
    slope = np.sqrt(nx * nx + ny * ny)
    slope = normalize_image_float(slope, percentile_clip=True)
    slope = apply_gamma_and_alpha(slope, alpha, gamma)
    return to_uint8_gray(slope)


def render_local_normal_deviation(
    nx,
    ny,
    nz,
    alpha,
    gamma,
    flip_x,
    flip_y,
    flip_z,
    local_radius: int,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
) -> np.ndarray:
    """
    Highlight local surface disturbance.

    Compares each normal to the locally averaged normal. Useful for faded
    inscriptions because broad object curvature is suppressed and small local
    changes are emphasized. The radius is interpreted in source pixels so
    previews and exports use comparable spatial neighborhoods.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)
    effective_radius = effective_radius_for_spacing(
        local_radius,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
    )
    bx, by, bz = box_blur_normals(nx, ny, nz, effective_radius)

    dot = nx * bx + ny * by + nz * bz
    dot = np.clip(dot, -1.0, 1.0)

    # angular deviation in radians
    deviation = np.arccos(dot).astype(np.float32)
    deviation = normalize_image_float(deviation, percentile_clip=True)
    deviation = apply_gamma_and_alpha(deviation, alpha, gamma)
    return to_uint8_gray(deviation)


def render_curvature_from_normals(
    nx,
    ny,
    nz,
    alpha,
    gamma,
    flip_x,
    flip_y,
    flip_z,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
) -> np.ndarray:
    """
    Curvature-like enhancement from normal divergence.

    Not a calibrated curvature measurement, but highly useful as a visual
    enhancement for incised or raised features.

    Approximation:
        curvature ≈ d(nx)/dx + d(ny)/dy
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)

    # np.gradient returns d/drow, d/dcol. Row corresponds to y, col to x.
    dnx_dy, dnx_dx = safe_gradient_2d(nx, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
    dny_dy, dny_dx = safe_gradient_2d(ny, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)

    curv = dnx_dx + dny_dy

    # Center zero curvature around mid-gray so positive/negative features differ.
    scale = np.percentile(np.abs(curv[np.isfinite(curv)]), 99.0)
    if scale <= 1e-8:
        out = np.full_like(curv, 0.5, dtype=np.float32)
    else:
        out = 0.5 + 0.5 * np.clip(curv / scale, -1.0, 1.0)

    out = apply_gamma_and_alpha(out, alpha, gamma)
    return to_uint8_gray(out)


def render_normal_gradient_magnitude(
    nx,
    ny,
    nz,
    alpha,
    gamma,
    flip_x,
    flip_y,
    flip_z,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
) -> np.ndarray:
    """
    Edge/feature strength from spatial changes in the normal field.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)

    nx_y, nx_x = safe_gradient_2d(nx, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
    ny_y, ny_x = safe_gradient_2d(ny, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
    nz_y, nz_x = safe_gradient_2d(nz, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)

    mag = np.sqrt(
        nx_x * nx_x + nx_y * nx_y
        + ny_x * ny_x + ny_y * ny_y
        + nz_x * nz_x + nz_y * nz_y
    )

    mag = normalize_image_float(mag, percentile_clip=True)
    mag = apply_gamma_and_alpha(mag, alpha, gamma)
    return to_uint8_gray(mag)


def render_mode_image(
    nx,
    ny,
    nz,
    alpha,
    mode: str,
    light: np.ndarray,
    ambient: float,
    gamma: float,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    multi_count: int,
    local_radius: int,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
) -> np.ndarray:
    """
    Main render dispatch function.

    Returns either:
      H x W uint8 grayscale
    or:
      H x W x 3 uint8 RGB
    """
    if mode == "Single light":
        return render_single_light(
            nx, ny, nz, alpha, light, ambient, gamma, flip_x, flip_y, flip_z
        )

    if mode in {
        "Mean multi-light",
        "Max multi-light",
        "Min multi-light",
        "Range multi-light",
        "Std-dev multi-light",
    }:
        return render_multi_light(
            nx,
            ny,
            nz,
            alpha,
            mode,
            light,
            ambient,
            gamma,
            flip_x,
            flip_y,
            flip_z,
            multi_count,
        )

    if mode == "RGB 3-light composite":
        return render_rgb_three_light_composite(
            nx, ny, nz, alpha, light, ambient, gamma, flip_x, flip_y, flip_z
        )

    if mode == "Slope from normals":
        return render_slope_from_normals(nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z)

    if mode == "Local normal deviation":
        return render_local_normal_deviation(
            nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z, local_radius, pixel_spacing_x, pixel_spacing_y
        )

    if mode == "Curvature from normals":
        return render_curvature_from_normals(nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z, pixel_spacing_x, pixel_spacing_y)

    if mode == "Normal gradient magnitude":
        return render_normal_gradient_magnitude(
            nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z, pixel_spacing_x, pixel_spacing_y
        )

    raise ValueError(f"Unknown render mode: {mode}")


# ---------------------------------------------------------------------------
# Hemisphere light selector widget
# ---------------------------------------------------------------------------

class LightHemisphereWidget(QtWidgets.QWidget):
    """
    Interactive upper-hemisphere light selector.

    Disk coordinates:
      center = overhead light, altitude 90°
      edge   = grazing light, altitude 0°
      top    = north/top light, azimuth 0°
      right  = east/right light, azimuth 90°
    """

    lightChanged = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(200, 200)
        self.setMaximumSize(240, 240)

        self.lx, self.ly, self.lz = az_alt_to_light_vector(315.0, 45.0)
        self.setMouseTracking(True)

    def set_from_az_alt(self, azimuth: float, altitude: float):
        self.lx, self.ly, self.lz = az_alt_to_light_vector(azimuth, altitude)
        self.update()
        self.lightChanged.emit(float(self.lx), float(self.ly), float(self.lz))

    def get_light(self) -> np.ndarray:
        return np.array([self.lx, self.ly, self.lz], dtype=np.float32)

    def get_az_alt(self):
        return light_vector_to_az_alt(self.lx, self.ly, self.lz)

    def _disk_geometry(self):
        w = self.width()
        h = self.height()

        margin = 24
        size = min(w, h) - 2 * margin
        radius = size / 2.0

        cx = w / 2.0
        cy = h / 2.0

        return cx, cy, radius

    @staticmethod
    def _event_position(event):
        """
        PyQt6 uses event.position(); older bindings may use event.pos().
        """
        if hasattr(event, "position"):
            return event.position()
        return event.pos()

    def _set_light_from_position(self, pos):
        cx, cy, radius = self._disk_geometry()

        dx = (pos.x() - cx) / radius
        dy = (pos.y() - cy) / radius

        r = math.sqrt(dx * dx + dy * dy)

        if r > 1.0:
            dx /= r
            dy /= r
            r = 1.0

        lx = dx
        ly = dy
        lz = math.sqrt(max(0.0, 1.0 - r * r))

        length = math.sqrt(lx * lx + ly * ly + lz * lz)
        self.lx = lx / length
        self.ly = ly / length
        self.lz = lz / length

        self.update()
        self.lightChanged.emit(float(self.lx), float(self.ly), float(self.lz))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._set_light_from_position(self._event_position(event))

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._set_light_from_position(self._event_position(event))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(245, 245, 245))

        cx, cy, radius = self._disk_geometry()

        rect = QtCore.QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        # Hemisphere disk background
        gradient = QtGui.QRadialGradient(QtCore.QPointF(cx, cy), radius)
        gradient.setColorAt(0.0, QtGui.QColor(245, 245, 245))
        gradient.setColorAt(0.65, QtGui.QColor(185, 185, 185))
        gradient.setColorAt(1.0, QtGui.QColor(95, 95, 95))

        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 2))
        painter.drawEllipse(rect)

        # Crosshair axes
        painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 1))
        painter.drawLine(QtCore.QPointF(cx - radius, cy), QtCore.QPointF(cx + radius, cy))
        painter.drawLine(QtCore.QPointF(cx, cy - radius), QtCore.QPointF(cx, cy + radius))

        # Cardinal labels
        painter.setPen(QtGui.QColor(30, 30, 30))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(
            QtCore.QRectF(cx - 15, cy - radius - 23, 30, 18),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "N",
        )
        painter.drawText(
            QtCore.QRectF(cx + radius + 5, cy - 9, 24, 18),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "E",
        )
        painter.drawText(
            QtCore.QRectF(cx - 15, cy + radius + 5, 30, 18),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "S",
        )
        painter.drawText(
            QtCore.QRectF(cx - radius - 29, cy - 9, 24, 18),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "W",
        )

        # Light position dot
        dot_x = cx + self.lx * radius
        dot_y = cy + self.ly * radius

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(dot_x, dot_y))

        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
        painter.drawEllipse(QtCore.QPointF(dot_x, dot_y), 8, 8)

        # Text below disk
        az, alt = self.get_az_alt()
        painter.setPen(QtGui.QColor(30, 30, 30))
        font = painter.font()
        font.setBold(False)
        painter.setFont(font)

        text = f"Az {az:.1f}°   Alt {alt:.1f}°"
        painter.drawText(
            QtCore.QRectF(0, self.height() - 24, self.width(), 20),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            text,
        )


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

class HillshadeViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        normal_path=None,
        output_path=None,
        max_preview_size=1800,
        initial_azimuth=315.0,
        initial_altitude=45.0,
        initial_ambient=0.15,
        initial_gamma=1.0,
    ):
        super().__init__()

        self.normal_path = Path(normal_path) if normal_path else None

        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = Path(default_output_for_input(normal_path))

        self.max_preview_size = int(max_preview_size)

        self.full_img = None
        self.large_height_source = None
        self.source_kind = "normal"
        self.height_stats = None

        self.full_nx = None
        self.full_ny = None
        self.full_nz = None
        self.full_alpha = None

        self.nx = None
        self.ny = None
        self.nz = None
        self.alpha = None

        self.preview_size = None
        self.full_size = None

        # Rotation is applied as a final image-space step to both preview and export.
        # It does not change light direction or normal/depth decoding.
        self.rotation_degrees = 0

        # Prevent feedback loops when the hemisphere and azimuth/altitude
        # sliders update each other.
        self._syncing_light_controls = False

        self._processing_message = ""

        self.setWindowTitle(APP_NAME)

        self._build_ui(
            initial_azimuth,
            initial_altitude,
            initial_ambient,
            initial_gamma,
        )

        self.update_timer = QtCore.QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_image)

        if self.normal_path:
            self.load_normal_map(self.normal_path)
        else:
            self.set_info("No image loaded. Click “Load image” to choose an RGB normal map or height/depth map. The interactive preview starts at 50%; exports use the full-resolution source.")

    def _build_ui(self, azimuth, altitude, ambient, gamma):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        outer_layout = QtWidgets.QVBoxLayout(central)

        main_layout = QtWidgets.QHBoxLayout()
        outer_layout.addLayout(main_layout, stretch=1)

        # Image view
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setMinimumSize(520, 360)

        self.view = self.graphics.addViewBox()
        self.view.setAspectLocked(True)

        # Make row 0 display at the top, like a normal image viewer.
        self.view.invertY(True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image_item)

        main_layout.addWidget(self.graphics, stretch=1)

        # Right control panel.
        #
        # The top part of the panel remains fixed: title/logo, file/export
        # buttons, status feedback, and render-mode selection. Scrolling starts
        # below the render-mode selector so the main application controls remain
        # visible even in smaller windows or fullscreen mode.
        controls_panel = QtWidgets.QWidget()
        controls_panel.setFixedWidth(390)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(6)

        header_row = QtWidgets.QHBoxLayout()

        header_text = QtWidgets.QWidget()
        header_text_layout = QtWidgets.QVBoxLayout(header_text)
        header_text_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout.setSpacing(2)

        title_label = QtWidgets.QLabel(APP_NAME)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_text_layout.addWidget(title_label)

        version_label = QtWidgets.QLabel(APP_VERSION)
        version_label.setStyleSheet("color: #666;")
        header_text_layout.addWidget(version_label)

        header_row.addWidget(header_text, stretch=1)

        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setFixedSize(92, 52)
        self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet(
            "QLabel { background: transparent; border: 0px; }"
        )
        header_row.addWidget(self.logo_label)

        controls_layout.addLayout(header_row)

        self.load_logo_from_file()

        # Fixed top buttons
        self.load_image_button = QtWidgets.QPushButton("Load image")
        self.save_button = QtWidgets.QPushButton("Save current render")
        self.save_16_button = QtWidgets.QPushButton("Save 16 single-light renders")
        self.reset_button = QtWidgets.QPushButton("Reset zoom")

        self.rotate_ccw_button = QtWidgets.QPushButton("Rotate ↺ 90°")
        self.rotate_cw_button = QtWidgets.QPushButton("Rotate ↻ 90°")

        self.invert_tones_button = QtWidgets.QPushButton("Invert tones: Off")
        self.invert_tones_button.setCheckable(True)

        controls_layout.addWidget(self.load_image_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.save_16_button)
        controls_layout.addWidget(self.reset_button)

        rotate_row = QtWidgets.QHBoxLayout()
        rotate_row.addWidget(self.rotate_ccw_button)
        rotate_row.addWidget(self.rotate_cw_button)
        controls_layout.addLayout(rotate_row)

        controls_layout.addWidget(self.invert_tones_button)

        self.processing_label = QtWidgets.QLabel("Ready")
        self.processing_label.setWordWrap(True)
        self.processing_label.setStyleSheet(
            "QLabel { "
            "background: #eef5ee; "
            "border: 1px solid #9ab99a; "
            "padding: 5px; "
            "color: #234b23; "
            "}"
        )
        controls_layout.addWidget(self.processing_label)

        # Input interpretation remains fixed above the scrolling section.
        controls_layout.addSpacing(6)
        controls_layout.addWidget(QtWidgets.QLabel("Input interpretation"))

        self.input_type_combo = QtWidgets.QComboBox()
        self.input_type_combo.addItems(["Auto detect", "RGB normal map", "Height/depth map"])
        controls_layout.addWidget(self.input_type_combo)

        # Render mode remains fixed above the scrolling section.
        controls_layout.addSpacing(6)
        controls_layout.addWidget(QtWidgets.QLabel("Render mode"))

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(RENDER_MODES)
        controls_layout.addWidget(self.mode_combo)

        # Scrollable lower controls
        lower_scroll = QtWidgets.QScrollArea()
        lower_scroll.setWidgetResizable(True)
        lower_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lower_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        lower_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        lower_controls = QtWidgets.QWidget()
        lower_controls_layout = QtWidgets.QVBoxLayout(lower_controls)
        lower_controls_layout.setContentsMargins(0, 8, 6, 0)
        lower_controls_layout.setSpacing(6)

        # Hemisphere light control
        lower_controls_layout.addWidget(QtWidgets.QLabel("Light direction"))

        self.light_widget = LightHemisphereWidget()
        self.light_widget.set_from_az_alt(azimuth, altitude)
        lower_controls_layout.addWidget(self.light_widget, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Azimuth / altitude sliders are kept in sync with the hemisphere.
        self.azimuth_slider = self._make_slider(0, 3600, int(azimuth * 10))
        self.altitude_slider = self._make_slider(0, 900, int(altitude * 10))

        lower_controls_layout.addWidget(QtWidgets.QLabel("Azimuth"))
        lower_controls_layout.addWidget(self.azimuth_slider)

        lower_controls_layout.addWidget(QtWidgets.QLabel("Altitude"))
        lower_controls_layout.addWidget(self.altitude_slider)

        # Sliders and numeric inputs
        self.ambient_slider = self._make_slider(0, 800, int(ambient * 1000))
        self.gamma_slider = self._make_slider(300, 2500, int(gamma * 1000))

        self.multi_count_input = self._make_spinbox(4, 32, 16)
        self.local_radius_input = self._make_spinbox(1, 150, 12)

        # Start at 50% preview resolution for a better performance/visibility
        # balance. Full-resolution export is unaffected.
        self.preview_percent_input = self._make_spinbox(1, 100, 50)

        lower_controls_layout.addSpacing(10)
        lower_controls_layout.addWidget(QtWidgets.QLabel("Ambient fill"))
        lower_controls_layout.addWidget(self.ambient_slider)

        lower_controls_layout.addWidget(QtWidgets.QLabel("Gamma / display contrast"))
        lower_controls_layout.addWidget(self.gamma_slider)

        lower_controls_layout.addWidget(QtWidgets.QLabel("Multi-light directions"))
        lower_controls_layout.addWidget(self.multi_count_input)

        lower_controls_layout.addWidget(QtWidgets.QLabel("Local radius"))
        lower_controls_layout.addWidget(self.local_radius_input)

        lower_controls_layout.addWidget(QtWidgets.QLabel("Preview resolution (%)"))
        lower_controls_layout.addWidget(self.preview_percent_input)

        # Height/depth-map conversion controls. These are ignored for RGB normal maps.
        self.height_strength_input = self._make_double_spinbox(0.0, 10000.0, 50.0, 0.5, 1)
        self.height_smooth_input = self._make_spinbox(0, 50, 0)
        self.invert_height_box = QtWidgets.QCheckBox("Invert height/depth before shading")

        lower_controls_layout.addSpacing(10)
        lower_controls_layout.addWidget(QtWidgets.QLabel("Height/depth relief strength"))
        lower_controls_layout.addWidget(self.height_strength_input)
        lower_controls_layout.addWidget(QtWidgets.QLabel("Height/depth smoothing radius"))
        lower_controls_layout.addWidget(self.height_smooth_input)
        lower_controls_layout.addWidget(self.invert_height_box)

        # Channel flips
        self.flip_x_box = QtWidgets.QCheckBox("Flip X / Red")
        self.flip_y_box = QtWidgets.QCheckBox("Flip Y / Green")
        self.flip_z_box = QtWidgets.QCheckBox("Flip Z / Blue")

        lower_controls_layout.addSpacing(10)
        lower_controls_layout.addWidget(self.flip_x_box)
        lower_controls_layout.addWidget(self.flip_y_box)
        lower_controls_layout.addWidget(self.flip_z_box)

        help_label = QtWidgets.QLabel(
            "Image viewer: mouse wheel zooms; left-drag pans.\n"
            "Rotate ↺/↻ changes both the preview and exported renders.\n"
            "Light control: drag the white dot or use azimuth/altitude sliders.\n"
            "Preview resolution changes interactive speed/detail only; exports use full-resolution source data.\n"
            "Very large single-band TIFF height/depth maps are read lazily with rasterio: preview is downsampled and TIFF export is tiled.\n"
            "For faint inscriptions, try Range multi-light, Std-dev multi-light, Local normal deviation, or Curvature from normals.\n"
            "Try Flip Y / Green if relief appears inverted; try Invert height/depth if raised/incised depth is reversed."
        )
        help_label.setWordWrap(True)

        lower_controls_layout.addSpacing(10)
        lower_controls_layout.addWidget(help_label)
        lower_controls_layout.addStretch()

        lower_scroll.setWidget(lower_controls)
        controls_layout.addWidget(lower_scroll, stretch=1)

        main_layout.addWidget(controls_panel)

        # Bottom information box
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(95)
        self.info_box.setMaximumHeight(130)
        self.info_box.setStyleSheet(
            "QTextEdit { "
            "background: #f7f7f7; "
            "border: 1px solid #aaa; "
            "padding: 6px; "
            "font-family: Consolas, monospace; "
            "font-size: 11px; "
            "}"
        )
        outer_layout.addWidget(self.info_box)

        self.colofon_bar = QtWidgets.QLabel(f"{APP_NAME} — {APP_VERSION} | {DEVELOPER_CREDIT}")
        self.colofon_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.colofon_bar.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.colofon_bar.setStyleSheet(
            "QLabel { "
            "background: #2b2b2b; "
            "color: #f2f2f2; "
            "padding: 5px; "
            "font-size: 11px; "
            "}"
        )
        outer_layout.addWidget(self.colofon_bar)

        # Signals
        self.light_widget.lightChanged.connect(self.on_light_widget_changed)
        self.azimuth_slider.valueChanged.connect(self.on_az_alt_slider_changed)
        self.altitude_slider.valueChanged.connect(self.on_az_alt_slider_changed)

        self.input_type_combo.currentTextChanged.connect(self.rebuild_preview_from_full_image)
        self.mode_combo.currentTextChanged.connect(self.request_update)
        self.ambient_slider.valueChanged.connect(self.request_update)
        self.gamma_slider.valueChanged.connect(self.request_update)
        self.multi_count_input.valueChanged.connect(self.request_update)
        self.local_radius_input.valueChanged.connect(self.request_update)
        self.preview_percent_input.valueChanged.connect(self.rebuild_preview_from_full_image)
        self.height_strength_input.valueChanged.connect(self.rebuild_preview_from_full_image)
        self.height_smooth_input.valueChanged.connect(self.rebuild_preview_from_full_image)
        self.invert_height_box.stateChanged.connect(self.rebuild_preview_from_full_image)

        self.flip_x_box.stateChanged.connect(self.request_update)
        self.flip_y_box.stateChanged.connect(self.request_update)
        self.flip_z_box.stateChanged.connect(self.request_update)

        self.load_image_button.clicked.connect(self.choose_image)
        self.save_button.clicked.connect(self.save_full_resolution)
        self.save_16_button.clicked.connect(self.save_16_single_light_renders)
        self.reset_button.clicked.connect(lambda: self.view.autoRange())
        self.rotate_ccw_button.clicked.connect(self.rotate_preview_export_counterclockwise)
        self.rotate_cw_button.clicked.connect(self.rotate_preview_export_clockwise)
        self.invert_tones_button.toggled.connect(self.toggle_invert_tones)

    def load_logo_from_file(self):
        """
        Load logo.png from the application folder and scale it to fit the
        top-right header area without distorting the aspect ratio.
        """
        logo_path = find_logo_path()

        if logo_path is None:
            self.logo_label.setText("")
            self.logo_label.setToolTip(f"No {LOGO_FILENAME} found next to the application file.")
            return

        pixmap = QtGui.QPixmap(str(logo_path))

        if pixmap.isNull():
            self.logo_label.setText("")
            self.logo_label.setToolTip(f"Could not load logo: {logo_path}")
            return

        scaled = pixmap.scaled(
            self.logo_label.width(),
            self.logo_label.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        self.logo_label.setPixmap(scaled)
        self.logo_label.setToolTip(str(logo_path))

    def set_busy(self, message: str | None = None):
        """
        Give visible feedback during longer operations such as full-resolution
        export and batch processing. This also forces the UI to repaint before
        the computation starts.
        """
        if message:
            self._processing_message = message
            self.processing_label.setText(message)
            self.processing_label.setStyleSheet(
                "QLabel { "
                "background: #fff3cd; "
                "border: 1px solid #d6b656; "
                "padding: 5px; "
                "color: #5b4500; "
                "font-weight: bold; "
                "}"
            )
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            self._processing_message = ""
            self.processing_label.setText("Ready")
            self.processing_label.setStyleSheet(
                "QLabel { "
                "background: #eef5ee; "
                "border: 1px solid #9ab99a; "
                "padding: 5px; "
                "color: #234b23; "
                "}"
            )
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass

        QtWidgets.QApplication.processEvents()

    @staticmethod
    def _make_slider(minimum, maximum, value):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(int(minimum))
        slider.setMaximum(int(maximum))
        slider.setValue(int(value))
        slider.setSingleStep(1)
        slider.setPageStep(10)
        return slider

    @staticmethod
    def _make_spinbox(minimum, maximum, value):
        spinbox = QtWidgets.QSpinBox()
        spinbox.setMinimum(int(minimum))
        spinbox.setMaximum(int(maximum))
        spinbox.setValue(int(value))
        spinbox.setSingleStep(1)
        spinbox.setKeyboardTracking(False)
        return spinbox

    @staticmethod
    def _make_double_spinbox(minimum, maximum, value, step=0.1, decimals=2):
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setMinimum(float(minimum))
        spinbox.setMaximum(float(maximum))
        spinbox.setValue(float(value))
        spinbox.setSingleStep(float(step))
        spinbox.setDecimals(int(decimals))
        spinbox.setKeyboardTracking(False)
        return spinbox

    def on_light_widget_changed(self, lx: float, ly: float, lz: float):
        """
        Update azimuth/altitude sliders when the hemisphere control changes.
        """
        if self._syncing_light_controls:
            self.request_update()
            return

        self._syncing_light_controls = True

        azimuth, altitude = light_vector_to_az_alt(lx, ly, lz)
        self.azimuth_slider.setValue(int(round(azimuth * 10)))
        self.altitude_slider.setValue(int(round(altitude * 10)))

        self._syncing_light_controls = False
        self.request_update()

    def on_az_alt_slider_changed(self, *args):
        """
        Update the hemisphere control when either azimuth or altitude slider changes.
        """
        if self._syncing_light_controls:
            return

        self._syncing_light_controls = True

        azimuth = self.azimuth_slider.value() / 10.0
        altitude = self.altitude_slider.value() / 10.0
        self.light_widget.set_from_az_alt(azimuth, altitude)

        self._syncing_light_controls = False
        self.request_update()

    def toggle_invert_tones(self, checked: bool):
        self.invert_tones_button.setText("Invert tones: On" if checked else "Invert tones: Off")
        self.request_update()

    def set_preview_export_rotation(self, rotation_degrees_clockwise: int | float):
        """
        Set the final image-space rotation used for both preview and export.
        """
        self.rotation_degrees = normalize_rotation_degrees(rotation_degrees_clockwise)

        if self.nx is not None:
            self.update_image()
            self.view.autoRange()
        else:
            self.update_info_box()

    def rotate_preview_export_clockwise(self):
        self.set_preview_export_rotation(self.rotation_degrees + 90)

    def rotate_preview_export_counterclockwise(self):
        self.set_preview_export_rotation(self.rotation_degrees - 90)

    def apply_preview_export_rotation(self, rendered: np.ndarray) -> np.ndarray:
        return rotate_rendered_array(rendered, self.rotation_degrees)

    def choose_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load RGB normal map or height/depth map",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All files (*)",
        )

        if filename:
            self.load_normal_map(filename)

    def selected_input_kind_for_image(self, img: Image.Image) -> str:
        """
        Resolve the input interpretation combo box to either "normal" or "height".
        """
        mode = self.input_type_combo.currentText() if hasattr(self, "input_type_combo") else "Auto detect"
        if mode == "RGB normal map":
            return "normal"
        if mode == "Height/depth map":
            return "height"
        return "height" if is_probably_heightmap(img) else "normal"

    def decode_source_to_normal_channels(self, img: Image.Image, pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0):
        """
        Decode either an RGB normal map or a height/depth map into normal
        channels used by the existing render modes.
        """
        kind = self.selected_input_kind_for_image(img)

        if kind == "height":
            nx, ny, nz, alpha, stats = heightmap_to_normal_channels(
                img,
                strength=float(self.height_strength_input.value()),
                invert_height=bool(self.invert_height_box.isChecked()),
                smooth_radius=int(self.height_smooth_input.value()),
                pixel_spacing_x=pixel_spacing_x,
                pixel_spacing_y=pixel_spacing_y,
            )
            return nx, ny, nz, alpha, kind, stats

        nx, ny, nz, alpha = decode_normal_map_from_pil(img)
        return nx, ny, nz, alpha, kind, None

    def get_full_processed_channels(self):
        """
        Return full-resolution normal channels for the classic in-memory path.
        Large height/depth rasters are exported through the tiled path instead.
        """
        if self.large_height_source is not None:
            raise RuntimeError(
                "Large height/depth rasters are not converted to full in-memory normal arrays. "
                "Use the tiled TIFF export path."
            )

        if self.full_img is None:
            return None, None, None, None

        if self.selected_input_kind_for_image(self.full_img) == "height":
            nx, ny, nz, alpha, kind, stats = self.decode_source_to_normal_channels(self.full_img)
            self.source_kind = kind
            self.height_stats = stats
            return nx, ny, nz, alpha

        if self.full_nx is None:
            nx, ny, nz, alpha, kind, stats = self.decode_source_to_normal_channels(self.full_img)
            self.full_nx, self.full_ny, self.full_nz, self.full_alpha = nx, ny, nz, alpha
            self.source_kind = kind
            self.height_stats = stats

        return self.full_nx, self.full_ny, self.full_nz, self.full_alpha

    def load_normal_map(self, path):
        path = Path(path)

        if not path.exists():
            self.set_info(f"Image not found: {path}")
            return

        # Close any previous lazy raster source.
        if self.large_height_source is not None:
            self.large_height_source.close()
            self.large_height_source = None

        self.full_img = None
        self.full_nx = self.full_ny = self.full_nz = self.full_alpha = None
        self.height_stats = None
        self.preview_pixel_spacing_x = 1.0
        self.preview_pixel_spacing_y = 1.0

        forced_mode = self.input_type_combo.currentText() if hasattr(self, "input_type_combo") else "Auto detect"
        large_source = try_open_large_heightmap_source(path, forced_mode=forced_mode)

        if large_source is not None:
            self.large_height_source = large_source
            self.normal_path = path
            self.output_path = Path(default_output_for_input(path))
            self.full_size = large_source.size
            self.source_kind = "height"

            try:
                self.set_busy("Reading downsampled height/depth statistics…")
                self.height_stats = large_source.estimate_stats()
            except Exception as exc:
                self.height_stats = None
                self.update_info_box(extra=f"Could not estimate global height stats; preview will use local stats. Error: {exc}")
            finally:
                self.set_busy(None)

            self.setWindowTitle(f"{APP_NAME} — {path.name}")
            self.rebuild_preview_from_full_image()
            self.view.autoRange()
            self.update_info_box(
                extra=(
                    f"Loaded large height/depth raster lazily: {path}\n"
                    f"Raster size: {large_source.width} × {large_source.height} px; "
                    f"dtype: {large_source.dtypes[0] if large_source.dtypes else 'unknown'}; "
                    f"rasterio backend: enabled."
                )
            )
            return

        try:
            full_img = Image.open(path)
            # For non-large-raster inputs we keep the older in-memory path.
            # Do not call this path for 4 GB height/depth rasters.
            full_img.load()
            full_img = full_img.copy()
        except Exception as exc:
            self.set_info(f"Could not load image:\n{path}\n\nError: {exc}")
            return

        self.full_img = full_img
        self.normal_path = path
        self.output_path = Path(default_output_for_input(path))
        self.full_size = full_img.size
        self.source_kind = self.selected_input_kind_for_image(full_img)

        # Predecode full-resolution RGB normal maps once. Height maps are
        # processed on demand because their conversion depends on UI controls.
        if self.source_kind == "normal":
            self.full_nx, self.full_ny, self.full_nz, self.full_alpha, _, _ = self.decode_source_to_normal_channels(full_img)

        self.setWindowTitle(f"{APP_NAME} — {path.name}")

        self.rebuild_preview_from_full_image()
        self.view.autoRange()
        self.update_info_box(extra=f"Loaded image: {path}")

    def rebuild_preview_from_full_image(self, *args):
        """
        Rebuild the interactive preview from the full-resolution source image.
        For large height/depth rasters, this reads a downsampled raster window
        with an automatic pixel cap instead of loading the whole file.
        """
        if self.large_height_source is not None:
            try:
                self.set_busy("Reading downsampled preview from large height/depth raster…")
                height, alpha = self.large_height_source.read_preview(
                    self.preview_percent_input.value(),
                    max_pixels=LARGE_RASTER_PREVIEW_MAX_PIXELS,
                )
                stats = self.height_stats or self.large_height_source.estimate_stats()
                self.nx, self.ny, self.nz, self.alpha, self.height_stats = height_array_to_normal_channels(
                    height,
                    alpha,
                    stats=stats,
                    strength=float(self.height_strength_input.value()),
                    invert_height=bool(self.invert_height_box.isChecked()),
                    smooth_radius=int(self.height_smooth_input.value()),
                    pixel_spacing_x=float(self.large_height_source.width) / max(1.0, float(height.shape[1])),
                    pixel_spacing_y=float(self.large_height_source.height) / max(1.0, float(height.shape[0])),
                )
                self.preview_size = (int(height.shape[1]), int(height.shape[0]))
                self.preview_pixel_spacing_x = float(self.large_height_source.width) / max(1.0, float(height.shape[1]))
                self.preview_pixel_spacing_y = float(self.large_height_source.height) / max(1.0, float(height.shape[0]))
                self.source_kind = "height"
                self.full_nx = self.full_ny = self.full_nz = self.full_alpha = None
            except Exception as exc:
                self.set_info(f"Could not build large-raster preview:\n{exc}")
                self.set_busy(None)
                return
            finally:
                self.set_busy(None)

            self.update_image()
            self.view.autoRange()
            return

        if self.full_img is None:
            return

        preview_img = make_preview_image_percent(
            self.full_img,
            self.max_preview_size,
            self.preview_percent_input.value(),
        )

        if self.full_size and self.selected_input_kind_for_image(self.full_img) == "height":
            self.preview_pixel_spacing_x = float(self.full_size[0]) / max(1.0, float(preview_img.size[0]))
            self.preview_pixel_spacing_y = float(self.full_size[1]) / max(1.0, float(preview_img.size[1]))
        else:
            self.preview_pixel_spacing_x = 1.0
            self.preview_pixel_spacing_y = 1.0

        self.nx, self.ny, self.nz, self.alpha, self.source_kind, self.height_stats = self.decode_source_to_normal_channels(
            preview_img,
            pixel_spacing_x=self.preview_pixel_spacing_x,
            pixel_spacing_y=self.preview_pixel_spacing_y,
        )

        # If the user switches interpretation back to RGB normal map, refresh
        # the cached full-resolution channels.
        if self.source_kind == "normal":
            self.full_nx, self.full_ny, self.full_nz, self.full_alpha, _, _ = self.decode_source_to_normal_channels(self.full_img)
        else:
            self.full_nx = self.full_ny = self.full_nz = self.full_alpha = None

        self.preview_size = preview_img.size
        self.update_image()
        self.view.autoRange()

    def get_params(self):
        light = self.light_widget.get_light()

        ambient = self.ambient_slider.value() / 1000.0
        gamma = self.gamma_slider.value() / 1000.0
        multi_count = int(self.multi_count_input.value())
        local_radius = int(self.local_radius_input.value())

        flip_x = self.flip_x_box.isChecked()
        flip_y = self.flip_y_box.isChecked()
        flip_z = self.flip_z_box.isChecked()

        invert_tones = self.invert_tones_button.isChecked()

        mode = self.mode_combo.currentText()
        input_interpretation = self.input_type_combo.currentText()
        height_strength = float(self.height_strength_input.value())
        height_smooth_radius = int(self.height_smooth_input.value())
        invert_height = bool(self.invert_height_box.isChecked())

        return {
            "mode": mode,
            "input_interpretation": input_interpretation,
            "source_kind": self.source_kind,
            "height_strength": height_strength,
            "height_smooth_radius": height_smooth_radius,
            "invert_height": invert_height,
            "rotation_degrees": int(self.rotation_degrees),
            "light": light,
            "ambient": ambient,
            "gamma": gamma,
            "multi_count": multi_count,
            "local_radius": local_radius,
            "preview_percent": int(self.preview_percent_input.value()),
            "flip_x": flip_x,
            "flip_y": flip_y,
            "flip_z": flip_z,
            "invert_tones": invert_tones,
        }

    def request_update(self, *args):
        if self.nx is None:
            return

        # Small debounce improves responsiveness during rapid dragging.
        self.update_timer.start(10)

    def render_current_preview(self) -> np.ndarray:
        params = self.get_params()

        # Large height/depth rasters use the same raw-render + final-normalize
        # path as tiled export. Earlier versions used the legacy preview renderer
        # here, which stretched Mean/Max/Min multi-light over the preview image,
        # while large export wrote direct [0,1] values. That made the preview and
        # exported TIFF look dramatically different. This keeps preview contrast
        # policy aligned with export contrast policy.
        if self.large_height_source is not None and self.source_kind == "height":
            raw = compute_normal_render_raw(
                self.nx,
                self.ny,
                self.nz,
                mode=params["mode"],
                light=params["light"],
                ambient=params["ambient"],
                flip_x=params["flip_x"],
                flip_y=params["flip_y"],
                flip_z=params["flip_z"],
                multi_count=params["multi_count"],
                local_radius=params["local_radius"],
                pixel_spacing_x=self.preview_pixel_spacing_x,
                pixel_spacing_y=self.preview_pixel_spacing_y,
            )

            if raw.ndim == 3:
                valid = np.isfinite(raw).all(axis=2) & (self.alpha > 0)
                values = raw[valid]
            else:
                valid = np.isfinite(raw) & (self.alpha > 0)
                values = raw[valid]

            render_norm = build_render_normalization_from_values(params["mode"], values)
            rendered = finalize_raw_render_to_uint8(
                raw,
                self.alpha,
                gamma=params["gamma"],
                render_norm=render_norm,
            )

            if params["invert_tones"]:
                rendered = invert_rendered_uint8(rendered)

            return self.apply_preview_export_rotation(rendered)

        rendered = render_mode_image(
            self.nx,
            self.ny,
            self.nz,
            self.alpha,
            mode=params["mode"],
            light=params["light"],
            ambient=params["ambient"],
            gamma=params["gamma"],
            flip_x=params["flip_x"],
            flip_y=params["flip_y"],
            flip_z=params["flip_z"],
            multi_count=params["multi_count"],
            local_radius=params["local_radius"],
            pixel_spacing_x=self.preview_pixel_spacing_x,
            pixel_spacing_y=self.preview_pixel_spacing_y,
        )

        if params["invert_tones"]:
            rendered = invert_rendered_uint8(rendered)

        return self.apply_preview_export_rotation(rendered)

    def render_current_fullres(self) -> np.ndarray:
        params = self.get_params()

        full_nx, full_ny, full_nz, full_alpha = self.get_full_processed_channels()

        rendered = render_mode_image(
            full_nx,
            full_ny,
            full_nz,
            full_alpha,
            mode=params["mode"],
            light=params["light"],
            ambient=params["ambient"],
            gamma=params["gamma"],
            flip_x=params["flip_x"],
            flip_y=params["flip_y"],
            flip_z=params["flip_z"],
            multi_count=params["multi_count"],
            local_radius=params["local_radius"],
            pixel_spacing_x=1.0,
            pixel_spacing_y=1.0,
        )

        if params["invert_tones"]:
            rendered = invert_rendered_uint8(rendered)

        return self.apply_preview_export_rotation(rendered)

    def update_image(self):
        if self.nx is None:
            return

        try:
            self.set_busy("Rendering preview…")
            rendered = self.render_current_preview()
        except Exception as exc:
            self.set_info(f"Render error:\n{exc}")
            self.set_busy(None)
            return

        self.image_item.setImage(
            rendered,
            autoLevels=False,
            levels=(0, 255),
        )

        self.update_info_box()
        self.set_busy(None)

    def update_info_box(self, extra=None):
        if self.normal_path is None:
            self.set_info(
                "No image loaded. Click “Load image” to choose an RGB normal map or height/depth map. "
                "The interactive preview starts at 50%; exports use the full-resolution source."
            )
            return

        params = self.get_params()
        light = params["light"]
        azimuth, altitude = light_vector_to_az_alt(light[0], light[1], light[2])

        full_w, full_h = self.full_size if self.full_size else (0, 0)
        prev_w, prev_h = self.preview_size if self.preview_size else (0, 0)

        lines = [
            f"Source image: {self.normal_path}",
            f"Input interpreted as: {params.get('source_kind', self.source_kind)}    UI setting: {params.get('input_interpretation')}",
            f"Render mode: {params['mode']}    Output default: {self.output_path}",
            f"Full-resolution source: {full_w} × {full_h} px    "
            f"Interactive preview data: {prev_w} × {prev_h} px at {params.get('preview_percent', self.preview_percent_input.value())}%",
            f"Export behaviour: Save current render and 16-light export use full-resolution source data with the same final rotation as the preview.",
            f"Preview/export rotation: {params.get('rotation_degrees', self.rotation_degrees)}° clockwise",
            f"Light vector: X {light[0]:+.3f}    Y {light[1]:+.3f}    Z {light[2]:+.3f}    "
            f"Azimuth: {azimuth:.1f}°    Altitude: {altitude:.1f}°",
            f"Display/settings: Ambient {params['ambient']:.2f}    Gamma {params['gamma']:.2f}    "
            f"Multi-light directions {params['multi_count']}    Local radius {params['local_radius']} px",
            f"Height/depth settings: Relief strength {params.get('height_strength', 0):.1f}    "
            f"Smoothing radius {params.get('height_smooth_radius', 0)} px    "
            f"Invert height/depth: {params.get('invert_height', False)}",
            f"Normal-channel flips: X/R {params['flip_x']}    Y/G {params['flip_y']}    Z/B {params['flip_z']}    "
            f"Invert tones: {params['invert_tones']}",
        ]

        if self.height_stats and params.get("source_kind") == "height":
            lines.append(
                "Height/depth stats: "
                f"min {self.height_stats.get('min')}    max {self.height_stats.get('max')}    "
                f"p1 {self.height_stats.get('p1')}    p99 {self.height_stats.get('p99')}"
            )

        if extra:
            lines.append(str(extra))

        self.set_info("\n".join(lines))

    def set_info(self, text):
        self.info_box.setPlainText(str(text))

    def build_processing_metadata(self, mode_override=None, light_override=None):
        params = self.get_params()

        mode = mode_override if mode_override is not None else params["mode"]
        light = light_override if light_override is not None else params["light"]

        azimuth, altitude = light_vector_to_az_alt(light[0], light[1], light[2])

        metadata = {
            "Software": APP_NAME,
            "SoftwareVersion": APP_VERSION,
            "Processing": "Render from RGB normal map or height/depth map",
            "RenderMode": mode,
            "InputInterpretation": params.get("input_interpretation"),
            "ResolvedSourceKind": self.source_kind,
            "Algorithm": (
                "RGB normal maps are decoded from [0,255] to [-1,+1]. Height/depth maps are "
                "robust-normalized, converted to normals from image gradients, and rendered using "
                "the selected normal-domain visualisation mode."
            ),
            "SourceImage": str(self.normal_path),
            "ExportedAtUTC": datetime.now(timezone.utc).isoformat(),

            "NormalMapConvention": "R=X, G=Y, B=Z; decoded from [0,255] to [-1,+1]",
            "NormalChannelR": "X",
            "NormalChannelG": "Y",
            "NormalChannelB": "Z",

            "LightVectorX": float(light[0]),
            "LightVectorY": float(light[1]),
            "LightVectorZ": float(light[2]),
            "AzimuthDegrees": float(azimuth),
            "AltitudeDegrees": float(altitude),

            "AmbientFill": float(params["ambient"]),
            "Gamma": float(params["gamma"]),
            "MultiLightDirections": int(params["multi_count"]),
            "LocalRadiusPixels": int(params["local_radius"]),
            "PreviewResolutionPercent": int(params.get("preview_percent", self.preview_percent_input.value())),
            "HeightReliefStrength": float(params.get("height_strength", 0.0)),
            "HeightSmoothRadiusPixels": int(params.get("height_smooth_radius", 0)),
            "InvertHeightDepth": bool(params.get("invert_height", False)),
            "HeightStats": self.height_stats,
            "OutputRotationDegreesClockwise": int(params.get("rotation_degrees", self.rotation_degrees)),
            "RotationAppliedAfterRendering": True,

            "FlipX_Red": bool(params["flip_x"]),
            "FlipY_Green": bool(params["flip_y"]),
            "FlipZ_Blue": bool(params["flip_z"]),
            "InvertTones": bool(params["invert_tones"]),

            "PreviewWidth": int(self.preview_size[0]) if self.preview_size else None,
            "PreviewHeight": int(self.preview_size[1]) if self.preview_size else None,
            "FullWidth": int(self.full_size[0]) if self.full_size else None,
            "FullHeight": int(self.full_size[1]) if self.full_size else None,
        }

        return metadata

    def save_image_with_metadata(
        self,
        rendered: np.ndarray,
        output_path: Path,
        mode_override=None,
        light_override=None,
    ):
        output_path = Path(output_path)

        if rendered.ndim == 2:
            img = Image.fromarray(rendered, mode="L")
        elif rendered.ndim == 3 and rendered.shape[2] == 3:
            img = Image.fromarray(rendered, mode="RGB")
        else:
            raise ValueError(f"Unsupported rendered image shape: {rendered.shape}")

        metadata = self.build_processing_metadata(
            mode_override=mode_override,
            light_override=light_override,
        )

        suffix = output_path.suffix.lower()

        if suffix == ".png":
            pnginfo = PngInfo()

            for key, value in metadata.items():
                pnginfo.add_text(str(key), str(value))

            pnginfo.add_text("ProcessingJSON", json.dumps(metadata, indent=2))
            img.save(output_path, pnginfo=pnginfo)

        elif suffix in [".tif", ".tiff"]:
            ifd = TiffImagePlugin.ImageFileDirectory_v2()
            ifd[270] = json.dumps(metadata, indent=2)  # ImageDescription
            ifd[305] = APP_NAME                       # Software
            img.save(output_path, tiffinfo=ifd)

        else:
            # Other formats may not preserve metadata reliably.
            # Save the image and write a sidecar JSON file.
            img.save(output_path)
            sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
            sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def render_large_heightmap_to_tiff_tiled(
        self,
        output_path: Path,
        mode_override=None,
        light_override=None,
        progress=None,
    ):
        """
        Full-resolution renderer for very large height/depth rasters.

        Output is always a tiled BigTIFF/GeoTIFF-compatible TIFF. Processing is
        performed block-by-block with a halo, so RAM use is roughly proportional
        to tile size rather than source size.
        """
        if self.large_height_source is None:
            raise RuntimeError("No large height/depth raster is loaded.")
        if rasterio is None:
            raise RuntimeError("rasterio is required for tiled large-raster export.")

        output_path = Path(output_path)
        params = self.get_params()
        mode = mode_override if mode_override is not None else params["mode"]
        light = light_override if light_override is not None else params["light"]
        rotation = normalize_rotation_degrees(params.get("rotation_degrees", self.rotation_degrees))
        output_width, output_height = rotated_dimensions(
            self.large_height_source.width,
            self.large_height_source.height,
            rotation,
        )

        stats = self.height_stats or self.large_height_source.estimate_stats()
        self.height_stats = stats

        render_norm = estimate_large_render_normalization(
            self.large_height_source,
            stats,
            params,
            mode,
            light,
        )

        # The RGB 3-light composite is the only RGB render mode; all others are grayscale.
        is_rgb = render_normalization_kind(mode) == "rgb_direct_01"
        count = 3 if is_rgb else 1

        profile = {
            "driver": "GTiff",
            "width": output_width,
            "height": output_height,
            "count": count,
            "dtype": "uint8",
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "YES",
        }

        # Copy georeferencing only when pixel orientation is unchanged.
        # A 90/180/270° image rotation changes the raster grid orientation;
        # keeping the original transform would georeference the exported image incorrectly.
        if rotation == 0:
            if self.large_height_source.transform is not None:
                profile["transform"] = self.large_height_source.transform
            if self.large_height_source.crs is not None:
                profile["crs"] = self.large_height_source.crs

        metadata = self.build_processing_metadata(mode_override=mode, light_override=light)
        metadata["OutputWidth"] = int(output_width)
        metadata["OutputHeight"] = int(output_height)
        metadata["GeoreferencingCopied"] = bool(rotation == 0)
        metadata["LargeRasterProcessing"] = "seam-safe tiled windowed export with halo and global render normalization"
        metadata["TileSize"] = LARGE_RASTER_TILE_SIZE
        metadata["TileHaloPixels"] = halo_for_large_height_render(
            mode,
            int(params.get("height_smooth_radius", 0)),
            int(params.get("local_radius", 0)),
        )
        metadata["TileRenderNormalization"] = json.dumps(render_norm)
        metadata["PreviewExportContrastPolicy"] = "Unified raw-render final-normalize path; Mean/Max/Min multi-light use global/sample-derived contrast for large heightmaps"

        tile_size = LARGE_RASTER_TILE_SIZE
        halo = int(metadata["TileHaloPixels"])
        total_tiles_x = int(math.ceil(self.large_height_source.width / tile_size))
        total_tiles_y = int(math.ceil(self.large_height_source.height / tile_size))
        total_tiles = total_tiles_x * total_tiles_y
        tile_index = 0

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.update_tags(
                Software=APP_NAME,
                ProcessingJSON=json.dumps(metadata, indent=2),
                RenderMode=str(mode),
            )

            for y in range(0, self.large_height_source.height, tile_size):
                h = min(tile_size, self.large_height_source.height - y)
                for x in range(0, self.large_height_source.width, tile_size):
                    w = min(tile_size, self.large_height_source.width - x)
                    tile_index += 1

                    if progress is not None:
                        if progress.wasCanceled():
                            raise RuntimeError("Export cancelled by user.")
                        progress.setLabelText(f"Rendering tile {tile_index} of {total_tiles}…")
                        progress.setValue(tile_index - 1)
                        QtWidgets.QApplication.processEvents()

                    height, alpha, crop = self.large_height_source.read_window_with_halo(x, y, w, h, halo)
                    rendered = render_height_tile_from_array_seam_safe(
                        height,
                        alpha,
                        stats,
                        params,
                        light_override=light,
                        mode_override=mode,
                        crop=crop,
                        render_norm=render_norm,
                    )
                    rendered = rotate_rendered_array(rendered, rotation)

                    dst_x, dst_y, dst_w, dst_h = rotated_window_for_source_tile(
                        x,
                        y,
                        w,
                        h,
                        self.large_height_source.width,
                        self.large_height_source.height,
                        rotation,
                    )
                    window = Window(dst_x, dst_y, dst_w, dst_h)

                    if is_rgb:
                        dst.write(np.ascontiguousarray(rendered[..., 0]), 1, window=window)
                        dst.write(np.ascontiguousarray(rendered[..., 1]), 2, window=window)
                        dst.write(np.ascontiguousarray(rendered[..., 2]), 3, window=window)
                    else:
                        dst.write(np.ascontiguousarray(rendered), 1, window=window)

            if progress is not None:
                progress.setValue(total_tiles)
                QtWidgets.QApplication.processEvents()

    def export_large_heightmap_tiled_with_dialog(self, requested_output_path: Path):
        if self.large_height_source is None:
            return False

        output_path = Path(requested_output_path)
        if output_path.suffix.lower() not in {".tif", ".tiff"}:
            output_path = output_path.with_suffix(".tif")
            self.output_path = output_path

        tile_size = LARGE_RASTER_TILE_SIZE
        total_tiles = int(math.ceil(self.large_height_source.width / tile_size)) * int(math.ceil(self.large_height_source.height / tile_size))

        progress = QtWidgets.QProgressDialog(
            "Preparing tiled BigTIFF export…",
            "Cancel",
            0,
            max(1, total_tiles),
            self,
        )
        progress.setWindowTitle("Processing large raster")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            self.set_busy("Rendering tiled full-resolution TIFF export…")
            self.update_info_box(extra="Large-raster export is tiled; the application has not finished yet.")
            self.render_large_heightmap_to_tiff_tiled(output_path, progress=progress)
        finally:
            progress.close()
            self.set_busy(None)

        self.update_info_box(extra=f"Saved tiled full-resolution TIFF render: {output_path}")
        QtWidgets.QMessageBox.information(
            self,
            "Save complete",
            f"Saved tiled full-resolution TIFF render:\n{output_path}",
        )
        print(f"Saved tiled full-resolution TIFF render: {output_path}")
        return True

    def save_full_resolution(self):
        if self.full_img is None and self.large_height_source is None:
            self.set_info("No image loaded. Cannot save.")
            return

        mode = self.mode_combo.currentText()
        suffix = f"_{safe_mode_name(mode)}"
        suggested = Path(default_output_for_input(self.normal_path, suffix=suffix))

        if self.large_height_source is not None:
            suggested = suggested.with_suffix(".tif")
            file_filter = "TIFF / BigTIFF image (*.tif *.tiff)"
            dialog_title = "Save tiled full-resolution render"
        else:
            file_filter = "PNG image (*.png);;TIFF image (*.tif *.tiff);;All files (*)"
            dialog_title = "Save full-resolution render"

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            dialog_title,
            str(suggested),
            file_filter,
        )

        if not filename:
            return

        self.output_path = Path(filename)

        if self.large_height_source is not None:
            try:
                self.export_large_heightmap_tiled_with_dialog(self.output_path)
            except Exception as exc:
                self.update_info_box(extra=f"Large-raster tiled save error: {exc}")
            return

        try:
            self.set_busy("Calculating and saving full-resolution render…")
            self.update_info_box(extra="Saving is in progress; the application has not finished yet.")
            rendered = self.render_current_fullres()
            self.save_image_with_metadata(rendered, self.output_path)
        except Exception as exc:
            self.update_info_box(extra=f"Save error: {exc}")
            return
        finally:
            self.set_busy(None)

        self.update_info_box(extra=f"Saved full-resolution render: {self.output_path}")
        QtWidgets.QMessageBox.information(
            self,
            "Save complete",
            f"Saved full-resolution render:\n{self.output_path}",
        )
        print(f"Saved full-resolution render: {self.output_path}")

    def save_16_single_light_renders_large_tiled(self):
        """
        Large-heightmap version of the 16-light export.
        Each full-resolution output is a tiled BigTIFF/GeoTIFF-compatible TIFF.
        A PNG contact sheet is generated from the interactive preview.
        """
        if self.large_height_source is None:
            self.set_info("No large height/depth raster loaded. Cannot run tiled batch export.")
            return

        default_dir = str(Path(self.normal_path).with_name(f"{Path(self.normal_path).stem}_16_hillshades_tiled"))

        outdir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose folder for 16 tiled TIFF hillshade renders",
            default_dir,
        )

        if not outdir:
            return

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        params = self.get_params()
        _, altitude = light_vector_to_az_alt(
            params["light"][0],
            params["light"][1],
            params["light"][2],
        )

        progress = QtWidgets.QProgressDialog(
            "Preparing 16 tiled single-light renders…",
            "Cancel",
            0,
            16,
            self,
        )
        progress.setWindowTitle("Processing large raster batch")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        contact_images = []
        labels = []

        try:
            self.set_busy("Processing 16 tiled single-light TIFF renders…")
            self.update_info_box(extra="Large-raster batch export is tiled; the application has not finished yet.")

            for i in range(16):
                if progress.wasCanceled():
                    self.update_info_box(extra="Batch export cancelled by user.")
                    return

                az = i * 22.5
                progress.setLabelText(f"Rendering full-resolution tiled TIFF {i + 1} of 16 at azimuth {az:.1f}°…")
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()

                light = az_alt_to_light_vector(az, altitude)
                az_label = f"{az:05.1f}".replace(".", "p")
                output_path = outdir / f"{Path(self.normal_path).stem}_hillshade_az_{az_label}_alt_{altitude:.1f}.tif"

                self.render_large_heightmap_to_tiff_tiled(
                    output_path,
                    mode_override="Single light",
                    light_override=light,
                    progress=None,
                )

                # Preview-sized contact sheet image from already built preview normals.
                rendered_preview = render_single_light(
                    self.nx,
                    self.ny,
                    self.nz,
                    self.alpha,
                    light,
                    params["ambient"],
                    params["gamma"],
                    params["flip_x"],
                    params["flip_y"],
                    params["flip_z"],
                )
                if params["invert_tones"]:
                    rendered_preview = invert_rendered_uint8(rendered_preview)

                rendered_preview = self.apply_preview_export_rotation(rendered_preview)

                thumb = Image.fromarray(rendered_preview, mode="L")
                thumb.thumbnail((360, 360), Image.Resampling.LANCZOS)
                contact_images.append(thumb.copy())
                labels.append(f"Az {az:.1f}°")

                progress.setValue(i + 1)
                QtWidgets.QApplication.processEvents()

            progress.setLabelText("Saving contact sheet…")
            QtWidgets.QApplication.processEvents()
            self.save_contact_sheet(contact_images, labels, outdir / f"{Path(self.normal_path).stem}_contact_sheet.png")

        except Exception as exc:
            self.update_info_box(extra=f"Large-raster batch export error: {exc}")
            return
        finally:
            progress.close()
            self.set_busy(None)

        self.update_info_box(extra=f"Saved 16 tiled TIFF hillshades to: {outdir}")
        QtWidgets.QMessageBox.information(
            self,
            "Batch export complete",
            f"Saved 16 tiled TIFF hillshades and contact sheet to:\n{outdir}",
        )
        print(f"Saved 16 tiled TIFF hillshades to: {outdir}")

    def save_16_single_light_renders(self):
        """
        Save 16 traditional raking-light hillshades at evenly spaced azimuths.
        Uses the current altitude, ambient, gamma, and flip settings.
        """
        if self.full_img is None and self.large_height_source is None:
            self.set_info("No image loaded. Cannot save.")
            return

        if self.large_height_source is not None:
            self.save_16_single_light_renders_large_tiled()
            return

        default_dir = str(Path(self.normal_path).with_name(f"{Path(self.normal_path).stem}_16_hillshades"))

        outdir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose folder for 16 hillshade renders",
            default_dir,
        )

        if not outdir:
            return

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        params = self.get_params()
        full_nx, full_ny, full_nz, full_alpha = self.get_full_processed_channels()
        _, altitude = light_vector_to_az_alt(
            params["light"][0],
            params["light"][1],
            params["light"][2],
        )

        progress = QtWidgets.QProgressDialog(
            "Preparing 16 single-light renders…",
            "Cancel",
            0,
            17,
            self,
        )
        progress.setWindowTitle("Processing")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        contact_images = []
        labels = []

        try:
            self.set_busy("Processing 16 single-light renders…")
            self.update_info_box(extra="Batch export is in progress; the application has not finished yet.")

            for i in range(16):
                if progress.wasCanceled():
                    self.update_info_box(extra="Batch export cancelled by user.")
                    return

                az = i * 22.5
                progress.setLabelText(f"Rendering image {i + 1} of 16 at azimuth {az:.1f}°…")
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()

                light = az_alt_to_light_vector(az, altitude)

                rendered = render_single_light(
                    full_nx,
                    full_ny,
                    full_nz,
                    full_alpha,
                    light,
                    params["ambient"],
                    params["gamma"],
                    params["flip_x"],
                    params["flip_y"],
                    params["flip_z"],
                )

                if params["invert_tones"]:
                    rendered = invert_rendered_uint8(rendered)

                rendered = self.apply_preview_export_rotation(rendered)

                az_label = f"{az:05.1f}".replace(".", "p")
                output_path = outdir / f"{Path(self.normal_path).stem}_hillshade_az_{az_label}_alt_{altitude:.1f}.png"

                progress.setLabelText(f"Saving image {i + 1} of 16…")
                QtWidgets.QApplication.processEvents()

                self.save_image_with_metadata(
                    rendered,
                    output_path,
                    mode_override="Single light",
                    light_override=light,
                )

                # Make small thumbnails for contact sheet
                thumb = Image.fromarray(rendered, mode="L")
                thumb.thumbnail((360, 360), Image.Resampling.LANCZOS)
                contact_images.append(thumb.copy())
                labels.append(f"Az {az:.1f}°")

                progress.setValue(i + 1)
                QtWidgets.QApplication.processEvents()

            if progress.wasCanceled():
                self.update_info_box(extra="Batch export cancelled by user.")
                return

            progress.setLabelText("Saving contact sheet…")
            progress.setValue(16)
            QtWidgets.QApplication.processEvents()

            self.save_contact_sheet(contact_images, labels, outdir / f"{Path(self.normal_path).stem}_contact_sheet.png")

            progress.setValue(17)

        except Exception as exc:
            self.update_info_box(extra=f"Batch export error: {exc}")
            return
        finally:
            progress.close()
            self.set_busy(None)

        self.update_info_box(extra=f"Saved 16 hillshades to: {outdir}")
        QtWidgets.QMessageBox.information(
            self,
            "Batch export complete",
            f"Saved 16 hillshades and contact sheet to:\n{outdir}",
        )
        print(f"Saved 16 hillshades to: {outdir}")

    def save_contact_sheet(self, pil_images, labels, out_path: Path, cols: int = 4):
        if not pil_images:
            return

        w, h = pil_images[0].size
        label_h = 28
        rows = int(math.ceil(len(pil_images) / cols))

        sheet = Image.new("L", (cols * w, rows * (h + label_h)), 255)
        draw = ImageDrawSafe(sheet)

        for i, img in enumerate(pil_images):
            col = i % cols
            row = i // cols

            x = col * w
            y = row * (h + label_h)

            sheet.paste(img.convert("L"), (x, y + label_h))
            draw.text((x + 8, y + 7), labels[i], fill=0)

        # Minimal metadata for contact sheet
        metadata = self.build_processing_metadata(mode_override="16 single-light contact sheet")

        pnginfo = PngInfo()
        pnginfo.add_text("ProcessingJSON", json.dumps(metadata, indent=2))
        pnginfo.add_text("Software", APP_NAME)
        pnginfo.add_text("RenderMode", "16 single-light contact sheet")

        sheet.save(out_path, pnginfo=pnginfo)


class ImageDrawSafe:
    """
    Tiny wrapper around PIL ImageDraw imported lazily.

    Keeps top-level imports compact and avoids font dependencies.
    """
    def __init__(self, image):
        from PIL import ImageDraw
        self.draw = ImageDraw.Draw(image)

    def text(self, *args, **kwargs):
        return self.draw.text(*args, **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast zoomable hemisphere-light viewer for RGB normal maps and height/depth maps."
    )

    parser.add_argument(
        "normal_map",
        nargs="?",
        default=None,
        help="Optional input RGB normal map or height/depth map",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Default output path when saving. If omitted, uses {input_filename}_hillshaded.png",
    )

    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=1800,
        help=(
            "Legacy option kept for compatibility. Preview size is controlled "
            "with the Preview resolution (%%) input in the UI."
        ),
    )

    parser.add_argument("--azimuth", type=float, default=315.0)
    parser.add_argument("--altitude", type=float, default=45.0)
    parser.add_argument("--ambient", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    app = QtWidgets.QApplication([])

    viewer = HillshadeViewer(
        normal_path=args.normal_map,
        output_path=args.output,
        max_preview_size=args.max_preview_size,
        initial_azimuth=args.azimuth,
        initial_altitude=args.altitude,
        initial_ambient=args.ambient,
        initial_gamma=args.gamma,
    )

    # Start in a moderate window size. The user can maximise or go fullscreen
    # afterwards; the side controls are scrollable so they remain accessible.
    viewer.resize(1180, 780)
    viewer.setMinimumSize(900, 620)
    viewer.show()

    app.exec()


if __name__ == "__main__":
    main()

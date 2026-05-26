#!/usr/bin/env python3
"""
Normal Map Hillshade Viewer
======================================

Interactive viewer for rendering low-relief heritage objects from RGB normal maps.

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
- multiple normal-map enhancement modes
- full-resolution export
- PNG/TIFF metadata embedding
- default output name: {input_filename}_hillshaded.png
- interactive preview resolution control, defaulting to 50%

Dependencies:
    pip install numpy pillow pyqtgraph PyQt6

Run:
    python main.py
    python main.py normal_map.png
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

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


APP_NAME = "Normal Map Hillshade Viewer"
APP_VERSION = "0.12 - 20260523"
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
# Normal-map loading and decoding
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
) -> np.ndarray:
    """
    Highlight local surface disturbance.

    Compares each normal to the locally averaged normal. Useful for faded
    inscriptions because broad object curvature is suppressed and small local
    changes are emphasized.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)
    bx, by, bz = box_blur_normals(nx, ny, nz, local_radius)

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
    dnx_dy, dnx_dx = np.gradient(nx)
    dny_dy, dny_dx = np.gradient(ny)

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
) -> np.ndarray:
    """
    Edge/feature strength from spatial changes in the normal field.
    """
    nx, ny, nz = corrected_channels(nx, ny, nz, flip_x, flip_y, flip_z)

    nx_y, nx_x = np.gradient(nx)
    ny_y, ny_x = np.gradient(ny)
    nz_y, nz_x = np.gradient(nz)

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
            nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z, local_radius
        )

    if mode == "Curvature from normals":
        return render_curvature_from_normals(nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z)

    if mode == "Normal gradient magnitude":
        return render_normal_gradient_magnitude(
            nx, ny, nz, alpha, gamma, flip_x, flip_y, flip_z
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
            self.set_info("No image loaded. Click “Load image” to choose an RGB normal map. The interactive preview starts at 50%; exports use the full-resolution normal map.")

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

        self.invert_tones_button = QtWidgets.QPushButton("Invert tones: Off")
        self.invert_tones_button.setCheckable(True)

        controls_layout.addWidget(self.load_image_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.save_16_button)
        controls_layout.addWidget(self.reset_button)
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
            "Light control: drag the white dot or use azimuth/altitude sliders.\n"
            "Preview resolution changes interactive speed/detail only; exports use the full-resolution normal map.\n"
            "For faint inscriptions, try Range multi-light, Std-dev multi-light, Local normal deviation, or Curvature from normals.\n"
            "Try Flip Y / Green if relief appears inverted."
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

        self.mode_combo.currentTextChanged.connect(self.request_update)
        self.ambient_slider.valueChanged.connect(self.request_update)
        self.gamma_slider.valueChanged.connect(self.request_update)
        self.multi_count_input.valueChanged.connect(self.request_update)
        self.local_radius_input.valueChanged.connect(self.request_update)
        self.preview_percent_input.valueChanged.connect(self.rebuild_preview_from_full_image)

        self.flip_x_box.stateChanged.connect(self.request_update)
        self.flip_y_box.stateChanged.connect(self.request_update)
        self.flip_z_box.stateChanged.connect(self.request_update)

        self.load_image_button.clicked.connect(self.choose_image)
        self.save_button.clicked.connect(self.save_full_resolution)
        self.save_16_button.clicked.connect(self.save_16_single_light_renders)
        self.reset_button.clicked.connect(lambda: self.view.autoRange())
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

    def choose_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load RGB normal map",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All files (*)",
        )

        if filename:
            self.load_normal_map(filename)

    def load_normal_map(self, path):
        path = Path(path)

        if not path.exists():
            self.set_info(f"Image not found: {path}")
            return

        try:
            full_img = Image.open(path).convert("RGBA")
        except Exception as exc:
            self.set_info(f"Could not load image:\n{path}\n\nError: {exc}")
            return

        self.full_img = full_img
        self.full_nx, self.full_ny, self.full_nz, self.full_alpha = decode_normal_map_from_pil(full_img)

        self.normal_path = path
        self.output_path = Path(default_output_for_input(path))
        self.full_size = full_img.size

        self.setWindowTitle(f"{APP_NAME} — {path.name}")

        self.rebuild_preview_from_full_image()
        self.view.autoRange()
        self.update_info_box(extra=f"Loaded image: {path}")

    def rebuild_preview_from_full_image(self, *args):
        """
        Rebuild the interactive preview from the full-resolution source image.
        This affects preview speed/resolution only. Full-resolution exports
        always use the original normal map.

        After rebuilding, the view is fitted to the new image extent so that
        changing the preview percentage, including switching to 100%, does not
        leave the user looking at only a cropped portion of the image.
        """
        if self.full_img is None:
            return

        preview_img = make_preview_image_percent(
            self.full_img,
            self.max_preview_size,
            self.preview_percent_input.value(),
        )

        self.nx, self.ny, self.nz, self.alpha = decode_normal_map_from_pil(preview_img)
        self.preview_size = preview_img.size
        self.update_image()

        # The preview image dimensions may have changed. Fit the whole image
        # into the viewport immediately; users can zoom in again afterwards.
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

        return {
            "mode": mode,
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
        )

        if params["invert_tones"]:
            rendered = invert_rendered_uint8(rendered)

        return rendered

    def render_current_fullres(self) -> np.ndarray:
        params = self.get_params()

        rendered = render_mode_image(
            self.full_nx,
            self.full_ny,
            self.full_nz,
            self.full_alpha,
            mode=params["mode"],
            light=params["light"],
            ambient=params["ambient"],
            gamma=params["gamma"],
            flip_x=params["flip_x"],
            flip_y=params["flip_y"],
            flip_z=params["flip_z"],
            multi_count=params["multi_count"],
            local_radius=params["local_radius"],
        )

        if params["invert_tones"]:
            rendered = invert_rendered_uint8(rendered)

        return rendered

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
                "No image loaded. Click “Load image” to choose an RGB normal map. "
                "The interactive preview starts at 50%; exports use the full-resolution normal map."
            )
            return

        params = self.get_params()
        light = params["light"]
        azimuth, altitude = light_vector_to_az_alt(light[0], light[1], light[2])

        full_w, full_h = self.full_size if self.full_size else (0, 0)
        prev_w, prev_h = self.preview_size if self.preview_size else (0, 0)

        lines = [
            f"Source normal map: {self.normal_path}",
            f"Render mode: {params['mode']}    Output default: {self.output_path}",
            f"Full-resolution source: {full_w} × {full_h} px    "
            f"Interactive preview data: {prev_w} × {prev_h} px at {params.get('preview_percent', self.preview_percent_input.value())}%",
            f"Export behaviour: Save current render and 16-light export use full-resolution normal-map data.",
            f"Light vector: X {light[0]:+.3f}    Y {light[1]:+.3f}    Z {light[2]:+.3f}    "
            f"Azimuth: {azimuth:.1f}°    Altitude: {altitude:.1f}°",
            f"Display/settings: Ambient {params['ambient']:.2f}    Gamma {params['gamma']:.2f}    "
            f"Multi-light directions {params['multi_count']}    Local radius {params['local_radius']} px",
            f"Normal-channel flips: X/R {params['flip_x']}    Y/G {params['flip_y']}    Z/B {params['flip_z']}    "
            f"Invert tones: {params['invert_tones']}",
        ]

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
            "Processing": "Render from RGB normal map",
            "RenderMode": mode,
            "Algorithm": (
                "RGB normal map decoded from [0,255] to [-1,+1], normalized per pixel, "
                "then rendered using the selected normal-domain visualisation mode."
            ),
            "SourceNormalMap": str(self.normal_path),
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

    def save_full_resolution(self):
        if self.full_nx is None:
            self.set_info("No image loaded. Cannot save.")
            return

        mode = self.mode_combo.currentText()
        suffix = f"_{safe_mode_name(mode)}"
        suggested = Path(default_output_for_input(self.normal_path, suffix=suffix))

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save full-resolution render",
            str(suggested),
            "PNG image (*.png);;TIFF image (*.tif *.tiff);;All files (*)",
        )

        if not filename:
            return

        self.output_path = Path(filename)

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

    def save_16_single_light_renders(self):
        """
        Save 16 traditional raking-light hillshades at evenly spaced azimuths.
        Uses the current altitude, ambient, gamma, and flip settings.
        """
        if self.full_nx is None:
            self.set_info("No image loaded. Cannot save.")
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
                    self.full_nx,
                    self.full_ny,
                    self.full_nz,
                    self.full_alpha,
                    light,
                    params["ambient"],
                    params["gamma"],
                    params["flip_x"],
                    params["flip_y"],
                    params["flip_z"],
                )

                if params["invert_tones"]:
                    rendered = invert_rendered_uint8(rendered)

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
        description="Fast zoomable hemisphere-light viewer for RGB normal maps."
    )

    parser.add_argument(
        "normal_map",
        nargs="?",
        default=None,
        help="Optional input RGB normal map",
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
            "with the Preview resolution (%) input in the UI."
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

#!/usr/bin/env python3

from pathlib import Path
import argparse
import math
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL import TiffImagePlugin
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import json
from datetime import datetime, timezone

APP_NAME = "Hemisphere Normal Map Hillshade Viewer"
APP_VERSION = "0.1 - 20260523"
LOGO = "logo.png"

pg.setConfigOptions(imageAxisOrder="row-major")

try:
    Signal = QtCore.Signal
except AttributeError:
    Signal = QtCore.pyqtSignal


def decode_normal_map_from_pil(img):
    """
    Decode RGB normal map from [0,255] to normalized [-1,1].

    Assumes:
      R = X
      G = Y
      B = Z
    """
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.float32)

    rgb = arr[..., :3] / 255.0
    alpha = arr[..., 3] / 255.0

    n = rgb * 2.0 - 1.0

    length = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.maximum(length, 1e-8)

    n = np.ascontiguousarray(n, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)

    return n[..., 0], n[..., 1], n[..., 2], alpha


def make_preview_image(img, max_preview_size):
    if max_preview_size <= 0:
        return img.copy()

    preview = img.copy()
    preview.thumbnail(
        (max_preview_size, max_preview_size),
        Image.Resampling.LANCZOS
    )
    return preview


def az_alt_to_light_vector(azimuth_deg, altitude_deg):
    """
    Azimuth:
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


def light_vector_to_az_alt(lx, ly, lz):
    """
    Convert image-coordinate light vector back to azimuth/altitude.
    """
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, lz))))
    azimuth = math.degrees(math.atan2(lx, -ly)) % 360.0
    return azimuth, altitude


def hillshade_from_channels(
    nx,
    ny,
    nz,
    alpha,
    light,
    ambient,
    gamma,
    flip_x=False,
    flip_y=False,
    flip_z=False,
):
    sx = -1.0 if flip_x else 1.0
    sy = -1.0 if flip_y else 1.0
    sz = -1.0 if flip_z else 1.0

    lx, ly, lz = light

    shade = (
        sx * nx * lx +
        sy * ny * ly +
        sz * nz * lz
    )

    shade = np.clip(shade, 0.0, 1.0)

    if ambient > 0:
        shade = ambient + (1.0 - ambient) * shade

    if gamma != 1.0:
        shade = np.clip(shade, 0.0, 1.0) ** gamma

    # Composite transparent/background areas onto white
    shade = shade * alpha + (1.0 - alpha) * 1.0

    return np.ascontiguousarray(np.clip(shade * 255.0, 0, 255).astype(np.uint8))


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

        self.setMinimumSize(230, 230)
        self.setMaximumSize(300, 300)

        self.lx, self.ly, self.lz = az_alt_to_light_vector(315.0, 45.0)

        self.setMouseTracking(True)

    def set_from_az_alt(self, azimuth, altitude):
        self.lx, self.ly, self.lz = az_alt_to_light_vector(azimuth, altitude)
        self.update()
        self.lightChanged.emit(float(self.lx), float(self.ly), float(self.lz))

    def get_light(self):
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
            self._set_light_from_position(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._set_light_from_position(event.position())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(245, 245, 245))

        cx, cy, radius = self._disk_geometry()

        rect = QtCore.QRectF(
            cx - radius,
            cy - radius,
            radius * 2,
            radius * 2
        )

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
        painter.drawLine(
            QtCore.QPointF(cx - radius, cy),
            QtCore.QPointF(cx + radius, cy)
        )
        painter.drawLine(
            QtCore.QPointF(cx, cy - radius),
            QtCore.QPointF(cx, cy + radius)
        )

        # Cardinal labels
        painter.setPen(QtGui.QColor(30, 30, 30))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(QtCore.QRectF(cx - 15, cy - radius - 23, 30, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "N")
        painter.drawText(QtCore.QRectF(cx + radius + 5, cy - 9, 24, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "E")
        painter.drawText(QtCore.QRectF(cx - 15, cy + radius + 5, 30, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "S")
        painter.drawText(QtCore.QRectF(cx - radius - 29, cy - 9, 24, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "W")

        # Light position dot
        dot_x = cx + self.lx * radius
        dot_y = cy + self.ly * radius

        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
        painter.drawEllipse(QtCore.QPointF(dot_x, dot_y), 8, 8)

        # Line from center to light position
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(dot_x, dot_y))

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
            text
        )


class HillshadeViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        normal_path=None,
        output_path="interactive_hillshade_fullres.png",
        logo_path=None,
        max_preview_size=1800,
        initial_azimuth=315.0,
        initial_altitude=45.0,
        initial_ambient=0.15,
        initial_gamma=1.0,
    ):
        super().__init__()

        self.normal_path = Path(normal_path) if normal_path else None
        self.output_path = Path(output_path)
        self.logo_path = Path(LOGO)
        self.max_preview_size = max_preview_size

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

        self.setWindowTitle("Hemisphere Normal Map Hillshade Viewer")

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
            self.set_info("No image loaded. Click “Load image” to choose an RGB normal map.")

    def _build_ui(self, azimuth, altitude, ambient, gamma):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        outer_layout = QtWidgets.QVBoxLayout(central)

        main_layout = QtWidgets.QHBoxLayout()
        outer_layout.addLayout(main_layout, stretch=1)

        # Image view
        self.graphics = pg.GraphicsLayoutWidget()
        self.view = self.graphics.addViewBox()
        self.view.setAspectLocked(True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image_item)

        main_layout.addWidget(self.graphics, stretch=1)

        # Right control panel
        controls = QtWidgets.QWidget()
        controls.setFixedWidth(350)
        controls_layout = QtWidgets.QVBoxLayout(controls)

        # Logo
        logo_row = QtWidgets.QHBoxLayout()
        logo_row.addStretch()

        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setFixedSize(220, 100)
        self.logo_label.setStyleSheet(
            "QLabel { border: 1px solid #999; background: #f5f5f5; }"
        )

        logo_row.addWidget(self.logo_label)
        controls_layout.addLayout(logo_row)

        # Buttons
        self.load_image_button = QtWidgets.QPushButton("Load image")
        self.save_button = QtWidgets.QPushButton("Save full-resolution PNG")
        self.reset_button = QtWidgets.QPushButton("Reset zoom")

        controls_layout.addWidget(self.load_image_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.reset_button)

        # Hemisphere light control
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QtWidgets.QLabel("Light direction hemisphere"))

        self.light_widget = LightHemisphereWidget()
        self.light_widget.set_from_az_alt(azimuth, altitude)
        controls_layout.addWidget(self.light_widget, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Ambient and gamma only
        self.ambient_slider = self._make_slider(0, 800, int(ambient * 1000))
        self.gamma_slider = self._make_slider(300, 2500, int(gamma * 1000))

        controls_layout.addSpacing(10)
        controls_layout.addWidget(QtWidgets.QLabel("Ambient fill"))
        controls_layout.addWidget(self.ambient_slider)

        controls_layout.addWidget(QtWidgets.QLabel("Gamma"))
        controls_layout.addWidget(self.gamma_slider)

        # Channel flips
        self.flip_x_box = QtWidgets.QCheckBox("Flip X / Red")
        self.flip_y_box = QtWidgets.QCheckBox("Flip Y / Green")
        self.flip_z_box = QtWidgets.QCheckBox("Flip Z / Blue")

        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.flip_x_box)
        controls_layout.addWidget(self.flip_y_box)
        controls_layout.addWidget(self.flip_z_box)

        help_label = QtWidgets.QLabel(
            "Image: mouse wheel = zoom, left-drag = pan.\n"
            "Light: drag the white dot inside the hemisphere.\n"
            "Center = overhead light. Edge = grazing light.\n"
            "Try Flip Y if relief looks inverted."
        )
        help_label.setWordWrap(True)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(help_label)
        controls_layout.addStretch()

        main_layout.addWidget(controls)

        # Bottom information box
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setFixedHeight(105)
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

        # Signals
        self.light_widget.lightChanged.connect(self.request_update)
        self.ambient_slider.valueChanged.connect(self.request_update)
        self.gamma_slider.valueChanged.connect(self.request_update)

        self.flip_x_box.stateChanged.connect(self.request_update)
        self.flip_y_box.stateChanged.connect(self.request_update)
        self.flip_z_box.stateChanged.connect(self.request_update)

        self.load_image_button.clicked.connect(self.choose_image)
        self.save_button.clicked.connect(self.save_full_resolution)
        self.reset_button.clicked.connect(lambda: self.view.autoRange())

    def _make_slider(self, minimum, maximum, value):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        return slider

    def set_logo(self, logo_path):
        logo_path = Path(logo_path)

        if not logo_path.exists():
            self.set_logo_placeholder()
            self.set_info(f"Logo not found: {logo_path}")
            return

        pixmap = QtGui.QPixmap(str(logo_path))

        if pixmap.isNull():
            self.set_logo_placeholder()
            self.set_info(f"Could not load logo: {logo_path}")
            return

        # Size of the logo display box in the UI
        target_width = self.logo_label.width()
        target_height = self.logo_label.height()

        # Scale down to fit, preserving aspect ratio.
        # Qt will not stretch the image beyond the target box.
        scaled = pixmap.scaled(
            target_width,
            target_height,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        self.logo_label.setPixmap(scaled)
        self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet(
            "QLabel { "
            "border: 0px; "
            "background: transparent; "
            "}"
        )

        self.logo_path = logo_path

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

        preview_img = make_preview_image(full_img, self.max_preview_size)

        self.full_nx, self.full_ny, self.full_nz, self.full_alpha = decode_normal_map_from_pil(full_img)
        self.nx, self.ny, self.nz, self.alpha = decode_normal_map_from_pil(preview_img)

        self.normal_path = path
        self.preview_size = preview_img.size
        self.full_size = full_img.size

        self.setWindowTitle(f"Hemisphere Normal Map Hillshade Viewer — {path.name}")

        self.update_image()
        self.view.autoRange()
        self.update_info_box(extra=f"Loaded image: {path}")

    def get_params(self):
        light = self.light_widget.get_light()

        ambient = self.ambient_slider.value() / 1000.0
        gamma = self.gamma_slider.value() / 1000.0

        flip_x = self.flip_x_box.isChecked()
        flip_y = self.flip_y_box.isChecked()
        flip_z = self.flip_z_box.isChecked()

        return light, ambient, gamma, flip_x, flip_y, flip_z

    def request_update(self, *args):
        if self.nx is None:
            return

        # Small debounce improves responsiveness during rapid dragging.
        self.update_timer.start(10)

    def update_image(self):
        if self.nx is None:
            return

        light, ambient, gamma, flip_x, flip_y, flip_z = self.get_params()

        shade = hillshade_from_channels(
            self.nx,
            self.ny,
            self.nz,
            self.alpha,
            light,
            ambient,
            gamma,
            flip_x,
            flip_y,
            flip_z,
        )

        self.image_item.setImage(
            shade,
            autoLevels=False,
            levels=(0, 255),
        )

        self.update_info_box()

    def update_info_box(self, extra=None):
        if self.normal_path is None:
            self.set_info("No image loaded. Click “Load image” to choose an RGB normal map.")
            return

        light, ambient, gamma, flip_x, flip_y, flip_z = self.get_params()
        azimuth, altitude = light_vector_to_az_alt(light[0], light[1], light[2])

        logo_text = str(self.logo_path) if self.logo_path else "No logo loaded"

        lines = [
            f"Image: {self.normal_path}",
            f"Preview size: {self.preview_size[0]} × {self.preview_size[1]} px    "
            f"Full size: {self.full_size[0]} × {self.full_size[1]} px",
            f"Light vector: X {light[0]:+.3f}    Y {light[1]:+.3f}    Z {light[2]:+.3f}",
            f"Azimuth: {azimuth:.1f}°    Altitude: {altitude:.1f}°    "
            f"Ambient: {ambient:.2f}    Gamma: {gamma:.2f}    "
            f"Flip X: {flip_x}    Flip Y: {flip_y}    Flip Z: {flip_z}",
            f"Logo: {logo_text}",
        ]

        if extra:
            lines.append(extra)

        self.set_info("\n".join(lines))

    def set_info(self, text):
        self.info_box.setPlainText(text)

    def build_processing_metadata(self):
        light, ambient, gamma, flip_x, flip_y, flip_z = self.get_params()
        azimuth, altitude = light_vector_to_az_alt(light[0], light[1], light[2])

        metadata = {
            "Software": APP_NAME,
            "SoftwareVersion": APP_VERSION,
            "Processing": "Direct hillshade from RGB normal map",
            "Algorithm": "Lambertian dot product between decoded normal vector and user-selected light vector",
            "SourceNormalMap": str(self.normal_path),
            "ExportedAtUTC": datetime.now(timezone.utc).isoformat(),

            "NormalMapConvention": "RGB normal map decoded from [0,255] to [-1,+1], then normalized per pixel",
            "NormalChannelR": "X",
            "NormalChannelG": "Y",
            "NormalChannelB": "Z",

            "LightVectorX": float(light[0]),
            "LightVectorY": float(light[1]),
            "LightVectorZ": float(light[2]),
            "AzimuthDegrees": float(azimuth),
            "AltitudeDegrees": float(altitude),

            "AmbientFill": float(ambient),
            "Gamma": float(gamma),

            "FlipX_Red": bool(flip_x),
            "FlipY_Green": bool(flip_y),
            "FlipZ_Blue": bool(flip_z),

            "PreviewWidth": int(self.preview_size[0]),
            "PreviewHeight": int(self.preview_size[1]),
            "FullWidth": int(self.full_size[0]),
            "FullHeight": int(self.full_size[1]),

            "LogoPath": str(self.logo_path),
        }

        return metadata

    def save_image_with_metadata(self, shade, output_path):
        output_path = Path(output_path)
        img = Image.fromarray(shade)
        metadata = self.build_processing_metadata()

        suffix = output_path.suffix.lower()

        if suffix == ".png":
            pnginfo = PngInfo()

            for key, value in metadata.items():
                pnginfo.add_text(str(key), str(value))

            pnginfo.add_text(
                "ProcessingJSON",
                json.dumps(metadata, indent=2)
            )

            img.save(output_path, pnginfo=pnginfo)

        elif suffix in [".tif", ".tiff"]:
            ifd = TiffImagePlugin.ImageFileDirectory_v2()
            ifd[270] = json.dumps(metadata, indent=2)  # ImageDescription
            ifd[305] = APP_NAME  # Software

            img.save(output_path, tiffinfo=ifd)

        else:
            # Other formats may not preserve metadata reliably.
            # Save the image and write a sidecar JSON file.
            img.save(output_path)

            sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
            sidecar_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8"
            )


    def save_full_resolution(self):
        if self.full_nx is None:
            self.set_info("No image loaded. Cannot save.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save full-resolution hillshade",
            str(self.output_path),
            "PNG image (*.png);;TIFF image (*.tif *.tiff);;All files (*)",
        )

        if not filename:
            return

        self.output_path = Path(filename)

        light, ambient, gamma, flip_x, flip_y, flip_z = self.get_params()

        shade = hillshade_from_channels(
            self.full_nx,
            self.full_ny,
            self.full_nz,
            self.full_alpha,
            light,
            ambient,
            gamma,
            flip_x,
            flip_y,
            flip_z,
        )

        self.save_image_with_metadata(shade, self.output_path)
        azimuth, altitude = light_vector_to_az_alt(light[0], light[1], light[2])

        self.update_info_box(extra=f"Saved full-resolution hillshade: {self.output_path}")

        print(f"Saved full-resolution hillshade: {self.output_path}")
        print(f"Light vector: X {light[0]:+.3f}, Y {light[1]:+.3f}, Z {light[2]:+.3f}")
        print(f"Azimuth: {azimuth:.1f}°")
        print(f"Altitude: {altitude:.1f}°")
        print(f"Ambient: {ambient:.2f}")
        print(f"Gamma: {gamma:.2f}")
        print(f"Flip X: {flip_x}")
        print(f"Flip Y: {flip_y}")
        print(f"Flip Z: {flip_z}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast zoomable hemisphere-light hillshade viewer for RGB normal maps."
    )

    parser.add_argument(
        "normal_map",
        nargs="?",
        default=None,
        help="Optional input RGB normal map"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="interactive_hillshade_fullres.png",
        help="Default output path when saving"
    )

    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=1800,
        help=(
            "Maximum width/height of interactive preview. "
            "Lower this for faster performance, e.g. 1200. "
            "Use 0 for full-resolution preview."
        )
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

    viewer.resize(1450, 980)
    viewer.show()

    app.exec()


if __name__ == "__main__":
    main()
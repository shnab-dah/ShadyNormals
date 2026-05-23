#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons


def load_normal_map(path):
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img).astype(np.float32)

    rgb = arr[..., :3] / 255.0
    alpha = arr[..., 3] / 255.0

    normals = rgb * 2.0 - 1.0

    length = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(length, 1e-8)

    return normals, alpha


def light_vector(azimuth_deg, altitude_deg):
    """
    Azimuth convention:
      0°   = light from top / north
      90°  = light from right / east
      180° = light from bottom / south
      270° = light from left / west

    Image coordinate convention:
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


def make_hillshade(normals, alpha, azimuth, altitude, ambient, gamma, flip_x, flip_y, flip_z):
    n = normals.copy()

    if flip_x:
        n[..., 0] *= -1
    if flip_y:
        n[..., 1] *= -1
    if flip_z:
        n[..., 2] *= -1

    light = light_vector(azimuth, altitude)

    shade = np.sum(n * light, axis=2)
    shade = np.clip(shade, 0.0, 1.0)

    shade = ambient + (1.0 - ambient) * shade
    shade = np.clip(shade, 0.0, 1.0)

    shade = shade ** gamma

    # Composite transparent areas onto white
    shade = shade * alpha + (1.0 - alpha) * 1.0

    return shade


def main():
    parser = argparse.ArgumentParser(
        description="Interactive hillshade viewer for RGB normal maps."
    )

    parser.add_argument("normal_map", help="Input RGB normal map")
    parser.add_argument(
        "-o",
        "--output",
        default="interactive_hillshade.png",
        help="Output filename when pressing Save"
    )

    parser.add_argument("--azimuth", type=float, default=315.0)
    parser.add_argument("--altitude", type=float, default=45.0)
    parser.add_argument("--ambient", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    normal_path = Path(args.normal_map)
    output_path = Path(args.output)

    normals, alpha = load_normal_map(normal_path)

    state = {
        "flip_x": False,
        "flip_y": False,
        "flip_z": False,
    }

    initial = make_hillshade(
        normals,
        alpha,
        args.azimuth,
        args.altitude,
        args.ambient,
        args.gamma,
        state["flip_x"],
        state["flip_y"],
        state["flip_z"],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.25, bottom=0.32)

    img_artist = ax.imshow(initial, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Interactive Normal Map Hillshade")
    ax.axis("off")

    # Slider axes
    ax_az = plt.axes([0.25, 0.22, 0.60, 0.03])
    ax_alt = plt.axes([0.25, 0.17, 0.60, 0.03])
    ax_amb = plt.axes([0.25, 0.12, 0.60, 0.03])
    ax_gamma = plt.axes([0.25, 0.07, 0.60, 0.03])

    slider_az = Slider(ax_az, "Azimuth", 0.0, 360.0, valinit=args.azimuth, valstep=1.0)
    slider_alt = Slider(ax_alt, "Altitude", 1.0, 89.0, valinit=args.altitude, valstep=1.0)
    slider_amb = Slider(ax_amb, "Ambient", 0.0, 0.8, valinit=args.ambient)
    slider_gamma = Slider(ax_gamma, "Gamma", 0.3, 2.5, valinit=args.gamma)

    # Checkboxes for channel flips
    ax_check = plt.axes([0.025, 0.52, 0.15, 0.13])
    checks = CheckButtons(
        ax_check,
        ["Flip X / Red", "Flip Y / Green", "Flip Z / Blue"],
        [False, False, False],
    )

    # Save button
    ax_save = plt.axes([0.025, 0.43, 0.15, 0.05])
    save_button = Button(ax_save, "Save PNG")

    def redraw(_=None):
        shade = make_hillshade(
            normals,
            alpha,
            slider_az.val,
            slider_alt.val,
            slider_amb.val,
            slider_gamma.val,
            state["flip_x"],
            state["flip_y"],
            state["flip_z"],
        )

        img_artist.set_data(shade)
        fig.canvas.draw_idle()

    def toggle_flip(label):
        if label == "Flip X / Red":
            state["flip_x"] = not state["flip_x"]
        elif label == "Flip Y / Green":
            state["flip_y"] = not state["flip_y"]
        elif label == "Flip Z / Blue":
            state["flip_z"] = not state["flip_z"]

        redraw()

    def save_current(_):
        shade = make_hillshade(
            normals,
            alpha,
            slider_az.val,
            slider_alt.val,
            slider_amb.val,
            slider_gamma.val,
            state["flip_x"],
            state["flip_y"],
            state["flip_z"],
        )

        out = (np.clip(shade, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out).save(output_path)

        print(f"Saved: {output_path}")
        print(f"Azimuth: {slider_az.val:.1f}°")
        print(f"Altitude: {slider_alt.val:.1f}°")
        print(f"Ambient: {slider_amb.val:.2f}")
        print(f"Gamma: {slider_gamma.val:.2f}")
        print(f"Flip X: {state['flip_x']}")
        print(f"Flip Y: {state['flip_y']}")
        print(f"Flip Z: {state['flip_z']}")

    slider_az.on_changed(redraw)
    slider_alt.on_changed(redraw)
    slider_amb.on_changed(redraw)
    slider_gamma.on_changed(redraw)
    checks.on_clicked(toggle_flip)
    save_button.on_clicked(save_current)

    plt.show()


if __name__ == "__main__":
    main()
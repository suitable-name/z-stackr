#!/usr/bin/env python3
"""Generates the application icons for z-stackr's public binaries and its web page.

Run from the workspace root:

    python scripts/make-icons.py

Writes:
    apps/stacker-gui/assets/icon.ico  + icon.png      (desktop app; icon.png is the Slint window icon)
    apps/stacker-cli/assets/icon.ico  + icon.png      (command line tool, same icon)
    apps/stacker-web/assets/favicon.png + favicon.svg (browser tab icon)

The generated files are committed. This script exists so the icons are reproducible
and adjustable rather than opaque binaries nobody can regenerate -- rerun it after
changing a color or the geometry below. Needs Pillow (`pip install pillow`).

# The subject

A focus stack: a dark rounded square holding four horizontal "focus planes" in green,
fading from faint at the bottom to fully sharp at the top (the one plane that is in
focus), over a light base line for the table the subject stands on. It is the same
motif as the project site's `icon.svg`, with the same colors and the same 64 unit
design grid, so the app, the site and the browser tab read as one product.

# Detail per size

PIL's draw primitives are not antialiased, so each frame is drawn at `SUPERSAMPLE` times
its size and shrunk with LANCZOS. Below 32 px the full design turns to mud (planes
2 px thick with 2 px gaps), so small sizes are simplified instead of blurred:

    >= 32 px   four planes + base line (the full motif)
    24 px      four planes, thicker, no base line
    16 px      three planes (25 / 62 / 100 percent), thicker, no base line

`favicon.svg` is written from the same constants, so it matches the 64 px design.
"""

import os

from PIL import Image, ImageDraw

# Rendered at this multiple of the target size, then downsampled with LANCZOS.
SUPERSAMPLE = 8

# Every size Windows actually asks for: Explorer's list/tile/icon views, the taskbar at
# each DPI scaling, and Alt-Tab.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

GROUND = "#111417"
PLANE = "#19c37d"
BASE = "#e9edef"

# The 64 unit design grid of the site icon: (y, opacity) of each 36 x 5 plane at x = 14.
FULL_PLANES = ((40, 0.25), (31, 0.5), (22, 0.75), (13, 1.0))
# Simplified sets, as (y, height, opacity), on the same grid.
MID_PLANES = ((40, 7, 0.25), (30, 7, 0.5), (20, 7, 0.75), (10, 7, 1.0))
SMALL_PLANES = ((38, 10, 0.25), (24, 10, 0.62), (10, 10, 1.0))
PLANE_X, PLANE_W = 14, 36


def _rgb(hex_str):
    h = hex_str.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _blend(fg, bg, alpha):
    return tuple(round(f * alpha + b * (1 - alpha)) for f, b in zip(fg, bg))


def _layout(size):
    """The planes `(y, height, opacity)` and whether to draw the base line at `size`."""
    if size >= 32:
        return tuple((y, 5, a) for y, a in FULL_PLANES), True
    if size >= 24:
        return MID_PLANES, False
    return SMALL_PLANES, False


def draw_icon(size):
    """Renders one square RGBA icon of `size` pixels."""
    s = size * SUPERSAMPLE
    k = s / 64  # design units -> pixels
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    ground = _rgb(GROUND)
    plane = _rgb(PLANE)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=12 * k, fill=ground + (255,))

    planes, base_line = _layout(size)
    for y, h, opacity in planes:
        # Blended onto the ground by hand: the plane is opaque in the image, so the ico
        # frame has no semi-transparent pixels inside the plate.
        d.rounded_rectangle(
            [PLANE_X * k, y * k, (PLANE_X + PLANE_W) * k, (y + h) * k],
            radius=h * k / 2,
            fill=_blend(plane, ground, opacity) + (255,),
        )
    if base_line:
        w = 3 * k
        y = 51 * k
        d.rounded_rectangle(
            [18 * k - w / 2, y - w / 2, 46 * k + w / 2, y + w / 2],
            radius=w / 2,
            fill=_rgb(BASE) + (255,),
        )

    return img.resize((size, size), Image.LANCZOS)


def write_svg(path):
    """The 64 unit motif as SVG, from the same constants as the raster frames."""
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">',
        "  <title>z-stackr</title>",
        f'  <rect width="64" height="64" rx="12" fill="{GROUND}"/>',
    ]
    for y, opacity in FULL_PLANES:
        op = "" if opacity == 1.0 else f' opacity="{opacity}"'
        lines.append(
            f'  <rect x="{PLANE_X}" y="{y}" width="{PLANE_W}" height="5" rx="2.5" fill="{PLANE}"{op}/>'
        )
    lines.append(
        f'  <path d="M18 51 H46" stroke="{BASE}" stroke-width="3" stroke-linecap="round"/>'
    )
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def build_app(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frames = [draw_icon(n) for n in ICO_SIZES]
    ico_path = os.path.join(out_dir, "icon.ico")
    # `append_images` is what makes this a genuine multi-resolution .ico -- each size is
    # its own rendered frame, not one bitmap Windows rescales badly.
    frames[-1].save(
        ico_path,
        format="ICO",
        sizes=[(n, n) for n in ICO_SIZES],
        append_images=frames[:-1],
    )
    png_path = os.path.join(out_dir, "icon.png")
    draw_icon(256).save(png_path, format="PNG")
    print(f"wrote {ico_path} and {png_path}")


def build_web(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "favicon.png")
    draw_icon(64).save(png_path, format="PNG")
    svg_path = os.path.join(out_dir, "favicon.svg")
    write_svg(svg_path)
    print(f"wrote {png_path} and {svg_path}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_app(os.path.join(root, "apps", "stacker-gui", "assets"))
    build_app(os.path.join(root, "apps", "stacker-cli", "assets"))
    build_web(os.path.join(root, "apps", "stacker-web", "assets"))

"""Draw the QuAIsar logo: mark + light/dark banners, as SVG and PNG.

The mark combines the three themes:
  * hexagon badge  -> cyber / security (a shield cell)
  * Bloch sphere with a tilted equator and state vector -> quantum
    (the equator doubles as a quasar's accretion disk, the vector as its jet)
  * nodes linked along the disk -> AI (a small neural network)

    python assets/make_logo.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon

OUT = Path(__file__).resolve().parent
plt.rcParams["svg.fonttype"] = "path"  # text becomes outlines: renders identically everywhere

CYAN, VIOLET, PINK, GOLD = "#22D3EE", "#8B5CF6", "#F472B6", "#FDE68A"
NAVY = "#0B1030"
GRAD = LinearSegmentedColormap.from_list("quaisar", [CYAN, VIOLET, PINK])
_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Segoe UI", "Inter", "Helvetica Neue", "Arial") if f in _fonts), "DejaVu Sans")

THEMES = {
    "light": {"text": "#0F172A", "muted": "#475569", "accent": "#0891B2"},
    "dark": {"text": "#F1F5F9", "muted": "#94A3B8", "accent": CYAN},
}


def diagonal(x, y):
    """Colour value from a top-left (cyan) to bottom-right (pink) diagonal, so every
    shape shares one light direction and closed outlines have no seam."""
    return np.clip((np.asarray(x) - np.asarray(y)) / (2 * 1.1 * np.sqrt(2)) * np.sqrt(2) + 0.5, 0, 1)


def gradient_line(ax, x, y, lw, alpha=1.0, cmap=GRAD, z=3, ls="solid"):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1), linewidths=lw, alpha=alpha, zorder=z,
                        capstyle="round", joinstyle="round", linestyles=ls)
    mid = segs.mean(axis=1)
    lc.set_array(diagonal(mid[:, 0], mid[:, 1]))
    ax.add_collection(lc)


def draw_mark(ax):
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal"); ax.axis("off")

    # hexagon badge (pointy top), filled navy with a gradient rim
    ang = np.deg2rad(90 + 60 * np.arange(7))
    hx, hy = 1.08 * np.cos(ang), 1.08 * np.sin(ang)
    ax.add_patch(Polygon(np.column_stack([hx, hy]), closed=True, fc=NAVY, ec="none", zorder=0,
                         joinstyle="round"))
    dense = np.linspace(0, 1, 6 * 40 + 1)
    seg = np.minimum((dense * 6).astype(int), 5)
    f = dense * 6 - seg
    rx = hx[seg] + (hx[seg + 1] - hx[seg]) * f
    ry = hy[seg] + (hy[seg + 1] - hy[seg]) * f
    gradient_line(ax, rx, ry, lw=7, z=1)

    # faint Bloch sphere outline and inner grid ring
    ax.add_patch(Circle((0, 0), 0.70, fill=False, ec="white", lw=1.6, alpha=0.22, zorder=1))
    ax.add_patch(Circle((0, 0), 0.70, fill=True, fc="#141A45", ec="none", alpha=0.55, zorder=0.5))

    tilt = np.deg2rad(-18)
    R = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])

    def disk(t, a=0.92, b=0.27):
        p = R @ np.vstack([a * np.cos(t), b * np.sin(t)])
        return p[0], p[1]

    # back half of the disk (behind the core): dim and thin
    t_back = np.linspace(0, np.pi, 120)
    x, y = disk(t_back)
    gradient_line(ax, x, y, lw=2.2, alpha=0.35, z=2)

    # jets / Bloch state vector along the disk's axis
    axis = R @ np.array([0, 1.0])
    for sign, length, alpha in ((1, 0.86, 1.0), (-1, 0.62, 0.45)):
        tip = sign * axis * length
        perp = np.array([-axis[1], axis[0]])
        for w, al in ((0.13, 0.10), (0.08, 0.18), (0.04, 0.45)):
            ax.add_patch(Polygon([perp * w, tip, -perp * w], closed=True,
                                 fc=CYAN if sign > 0 else VIOLET, ec="none", alpha=al * alpha, zorder=2.5))
        ax.plot([0, tip[0] * 0.9], [0, tip[1] * 0.9], color="white", lw=2.2, alpha=0.9 * alpha,
                solid_capstyle="round", zorder=4)
    tip = axis * 0.86
    perp = np.array([-axis[1], axis[0]])
    head = [tip + axis * 0.10, tip + perp * 0.075 - axis * 0.04, tip - perp * 0.075 - axis * 0.04]
    ax.add_patch(Polygon(head, closed=True, fc="white", ec="none", zorder=4.5))

    # glowing core
    for r, al in zip(np.linspace(0.36, 0.06, 14), np.linspace(0.03, 0.22, 14)):
        ax.add_patch(Circle((0, 0), r, fc=GOLD, ec="none", alpha=al, zorder=3))
    ax.add_patch(Circle((0, 0), 0.075, fc="white", ec="none", zorder=5))

    # front half of the disk (in front of the core): bright and thick
    t_front = np.linspace(np.pi, 2 * np.pi, 160)
    x, y = disk(t_front)
    gradient_line(ax, x, y, lw=5.5, z=6)

    # AI: nodes on the disk, linked to each other and to the core
    # end nodes sit exactly where the front and back halves meet, hiding the joins
    tn = np.array([1.0, 1.36, 1.68, 2.0]) * np.pi
    nx, ny = disk(tn)
    colors = [GRAD(v) for v in diagonal(nx, ny)]
    for i in range(len(tn)):
        ax.plot([0, nx[i]], [0, ny[i]], color="white", lw=0.9, alpha=0.30, zorder=5.5)
    for x0, y0, c in zip(nx, ny, colors):
        ax.add_patch(Circle((x0, y0), 0.075, fc=NAVY, ec="none", zorder=6.5))
        ax.add_patch(Circle((x0, y0), 0.058, fc=c, ec="white", lw=1.4, zorder=7))


def save(fig, stem, **kw):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", transparent=True, dpi=220,
                    bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)


def mark():
    fig = plt.figure(figsize=(4, 4))
    draw_mark(fig.add_axes([0, 0, 1, 1]))
    save(fig, "logo-mark")


def banner(theme):
    c = THEMES[theme]
    fig = plt.figure(figsize=(7.4, 2.6))
    draw_mark(fig.add_axes([0.0, 0.02, 0.337, 0.96]))
    ax = fig.add_axes([0.325, 0, 0.675, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # wordmark: Qu | AI | sar, with "AI" in the accent colour
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x = 0.02
    for part, color in (("Qu", c["text"]), ("AI", c["accent"]), ("sar", c["text"])):
        t = ax.text(x, 0.56, part, fontsize=66, fontfamily=FONT, fontweight="bold", color=color, va="baseline")
        bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        x = bb.x1 + 0.004
    ax.text(0.024, 0.33, "Quantum × Cyber × AI research lab", fontsize=19,
            fontfamily=FONT, color=c["muted"], va="baseline")
    ax.text(0.024, 0.12, "Created with Claude Code", fontsize=13, fontfamily=FONT,
            color=c["muted"], va="baseline", style="italic")
    save(fig, f"banner-{theme}")


if __name__ == "__main__":
    mark()
    for theme in THEMES:
        banner(theme)
    print("Wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg")) + sorted(p.name for p in OUT.glob("*.png"))))

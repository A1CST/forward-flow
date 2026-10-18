"""Endless cartoon street backdrop for demo videos (Claude 2026-10-17). Deterministic tiles: buildings, windows, doors,
lamp posts, trees, sidewalk and curb, scrolled by a world offset in px so a walk reads as moving down the street.
  street(offset_px, size=256, seed=0) -> RGB uint8 (size, size, 3)
"""
import numpy as np
from PIL import Image, ImageDraw

TILE = 150


def _rng(seed, k): return np.random.default_rng((seed * 1_000_003 + k) & 0xFFFFFFFF)


def street(offset_px, size=256, seed=0, ground_px=None):
    im = Image.new("RGB", (size, size), (158, 206, 236)); d = ImageDraw.Draw(im)
    ground = int(ground_px if ground_px is not None else size * .80)
    d.rectangle((0, int(size * .55), size, ground), fill=(176, 214, 238))                    # haze band
    k0 = int(np.floor(offset_px / TILE)) - 1
    for k in range(k0, k0 + size // TILE + 3):                                                  # buildings (far layer, .6 parallax)
        r = _rng(seed, k); x0 = k * TILE - offset_px * 1.0
        w = int(r.integers(95, 140)); h = int(r.integers(70, 150)); col = tuple(int(v) for v in r.integers(150, 235, 3))
        bx = int(x0 + (TILE - w) / 2); d.rectangle((bx, ground - h, bx + w, ground), fill=col, outline=(70, 66, 80), width=2)
        d.rectangle((bx - 3, ground - h - 6, bx + w + 3, ground - h), fill=tuple(max(0, c - 50) for c in col), outline=(70, 66, 80), width=2)
        for wy in range(ground - h + 14, ground - 30, 26):
            for wx in range(bx + 10, bx + w - 16, 24):
                d.rectangle((wx, wy, wx + 12, wy + 14), fill=(120, 176, 205), outline=(70, 66, 80), width=1)
        dx = bx + w // 2 - 9; d.rectangle((dx, ground - 30, dx + 18, ground), fill=tuple(max(0, c - 70) for c in col), outline=(70, 66, 80), width=2)
    for k in range(int(np.floor(offset_px / 90)) - 1, int(np.floor(offset_px / 90)) + size // 90 + 3):   # lamps
        x = int(k * 90 - offset_px) + 20; d.rectangle((x - 2, ground - 95, x + 2, ground), fill=(60, 70, 86))
        d.ellipse((x - 8, ground - 104, x + 8, ground - 88), fill=(255, 226, 140), outline=(60, 70, 86), width=2)
    for k in range(int(np.floor(offset_px / 210)) - 1, int(np.floor(offset_px / 210)) + size // 210 + 3):  # trees
        r = _rng(seed + 7, k); x = int(k * 210 - offset_px) + int(r.integers(60, 150))
        d.rectangle((x - 4, ground - 50, x + 4, ground), fill=(110, 78, 56))
        d.ellipse((x - 26, ground - 92, x + 26, ground - 40), fill=(70, 146, 92), outline=(50, 96, 66), width=2)
    d.rectangle((0, ground, size, size), fill=(186, 186, 194)); d.rectangle((0, ground, size, ground + 4), fill=(140, 140, 150))
    for k in range(int(np.floor(offset_px / 40)) - 1, int(np.floor(offset_px / 40)) + size // 40 + 3):   # paving joints
        x = int(k * 40 - offset_px); d.line((x, ground + 4, x - 10, size), fill=(160, 160, 170), width=1)
    d.rectangle((0, min(size - 4, ground + 26), size, size), fill=(92, 94, 104))                           # road edge
    return np.asarray(im)

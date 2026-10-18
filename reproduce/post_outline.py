"""Cartoon outlines as a post-process on the Blender pack (Claude 2026-10-17): Freestyle made rendering ~7x slower.
Draws a dark line on the silhouette (1 px outside the alpha) and on every boundary between label regions inside the
figure, in place, for every rgba/<t>.png (+ its labels/<t>.png) and every ref.png (+ ref_labels.png). Idempotent marker file.

  python character/post_outline.py --pack character/packs/human_v1 [--workers 32]
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

LINE = np.array([26, 26, 36], np.float32)


def outline(rgba_p, lab_p):
    x = np.asarray(Image.open(rgba_p).convert("RGBA")).astype(np.float32); lab = np.asarray(Image.open(lab_p).convert("L")).astype(np.int16)
    a = x[..., 3] > 127; q = (lab + 20) // 40                                    # region index (grey levels 40,80,..; +-1 tolerated)
    def nb(m):
        o = np.zeros_like(m)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
        return o
    outer = nb(a) & ~a                                                           # 1 px ring outside the silhouette
    inner = np.zeros_like(a)
    for dy, dx in ((0, 1), (1, 0)):
        inner |= a & np.roll(np.roll(a, dy, 0), dx, 1) & (q != np.roll(np.roll(q, dy, 0), dx, 1))
    x[outer, :3] = LINE; x[outer, 3] = 255
    x[inner, :3] = x[inner, :3] * .25 + LINE * .75
    Image.fromarray(x.clip(0, 255).astype(np.uint8)).save(rgba_p)
    return 1


def job(args):
    return outline(*args)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", type=Path, required=True); ap.add_argument("--workers", type=int, default=32); a = ap.parse_args()
    mark = a.pack / ".outlined"
    if mark.exists(): print("already outlined"); return
    pairs = [(p, p.parent.parent / "labels" / p.name) for p in a.pack.glob("*/*/rgba/*.png")]
    pairs += [(p, p.with_name("ref_labels.png")) for p in a.pack.glob("*/ref.png")]
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(job, pairs, chunksize=64))
    mark.write_text(str(n)); print("outlined", n)


if __name__ == "__main__":
    main()

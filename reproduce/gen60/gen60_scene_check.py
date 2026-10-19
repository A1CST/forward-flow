"""GEN60 scene continuity check (Payton 1561554807041425430 'some of the buildings glitch'; Astra 1561554916793651311 before/after).
Occlusion truth = an exact per-pixel ray cast against every building box (and stepped-roof box); painter = the renderer's draw
order with flat ID colours. A pixel where they disagree is a draw-order glitch. Gable triangles are skipped in both (no box).
  python render/gen60_scene_check.py --order centroid|nearest|exact --every 6 --size 128 --out <json>"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_scene_v4 as V4


def boxes(world):
    out = []
    for i, b in enumerate(world, 1):
        lo, hi = (b["x0"], 0.0, min(b["zf"], b["zf"] + b["side"] * b["d"])), (b["x1"], b["h"], max(b["zf"], b["zf"] + b["side"] * b["d"]))
        out.append((i, lo, hi))
        if b["roof"] == "step":
            inset = (b["x1"] - b["x0"]) * 0.2; zf = b["zf"] + b["side"] * 1.5; zb = zf + b["side"] * (b["d"] - 2)
            out.append((i, (b["x0"] + inset, b["h"], min(zf, zb)), (b["x1"] - inset, b["h"] + 3, max(zf, zb))))
    return out


def ray_ids(world, cam, S):
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float64) + 0.5; f = cam.f * S / V4.W
    R, U, F = (np.array(v) for v in (cam.right, cam.up, cam.fwd))
    d = F[None, None] + R[None, None] * ((xs - S / 2) / f)[..., None] + U[None, None] * ((S / 2 - ys) / f)[..., None]
    O = np.array(cam.C); best = np.full((S, S), np.inf); ids = np.zeros((S, S), np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, lo, hi in boxes(world):
            t0 = (np.array(lo) - O) / d; t1 = (np.array(hi) - O) / d
            tn = np.nanmax(np.minimum(t0, t1), -1); tf = np.nanmin(np.maximum(t0, t1), -1)
            hit = (tf >= tn) & (tf > 0); tt = np.where(tn > 0, tn, tf)
            fz = tt * np.einsum("ijk,k->ij", d, F)                       # camera-space depth of the hit (NEAR plane test)
            hit &= fz >= V4.NEAR; upd = hit & (tt < best); best[upd] = tt[upd]; ids[upd] = i
    return ids


def painter_ids(world, cam, S, order):
    img = Image.new("I", (V4.W, V4.H), 0); dr = ImageDraw.Draw(img); idx = {b["id"]: i for i, b in enumerate(world, 1)}
    vis = [b for b in world if V4.visible(cam, b)]
    if order == "nearest": key = lambda b: V4.nearest_dist(b, cam.C)
    else: key = lambda b: cam.cam(((b["x0"] + b["x1"]) / 2, b["h"] / 2, b["zf"] + b["side"] * b["d"] / 2))[2]     # v4.0 (before)
    seq = [it[1] for it in V4.draw_order([("b", b, V4.aabb(b)) for b in vis], cam.C)] if order == "exact" else sorted(vis, key=lambda b: -key(b))
    for b in seq:
        for name, poly, _ in V4.box_faces(b, cam):
            if name != "gable": dr.polygon(poly, fill=idx[b["id"]])
    return np.asarray(img.resize((S, S), Image.NEAREST), np.int32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--order", default="nearest"); ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--out", required=True)
    ap.add_argument("--path", default=str(Path(__file__).resolve().parent / "gen60" / "path60.json")); a = ap.parse_args()
    world = V4.make_street(); fr = json.loads(Path(a.path).read_text())["frames"][::a.every]; rows = []
    for f in fr:
        cam = V4.Cam(tuple(f["C"]), tuple(f["L"]), V4.F_PX * f.get("zoom", 1.0))
        r, p = ray_ids(world, cam, a.size), painter_ids(world, cam, a.size, a.order)
        bad = (r != p) & (r > 0) & (p > 0)                                       # wrong building in front (ignores silhouette edges vs sky)
        pad = np.pad(r, 1, mode="edge"); same = np.ones_like(r, bool)            # interior = whole 3x3 ray neighbourhood is one id (not an edge)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1): same &= pad[1 + dy:1 + dy + a.size, 1 + dx:1 + dx + a.size] == r
        rows.append({"f": f["f"], "t": f["t"], "wrong_building_px_pct": round(100 * float(bad.mean()), 3),
                     "edge_px_pct": round(100 * float(((r > 0) != (p > 0)).mean()), 3), "interior_wrong_px": int((bad & same).sum())})
    w = [x["wrong_building_px_pct"] for x in rows]
    res = {"order": a.order, "frames": len(rows), "size": a.size, "mean_wrong_pct": round(float(np.mean(w)), 4), "max_wrong_pct": max(w),
           "frames_over_0_5pct": sum(x > 0.5 for x in w), "interior_wrong_px_total": sum(x["interior_wrong_px"] for x in rows), "frames_with_interior_wrong": sum(x["interior_wrong_px"] > 0 for x in rows), "worst": sorted(rows, key=lambda x: -x["wrong_building_px_pct"])[:8], "rows": rows}
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()

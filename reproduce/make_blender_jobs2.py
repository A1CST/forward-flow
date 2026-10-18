"""Blender pack jobs v2: scripted multi-action clips (Claude 2026-10-17; Payton 1560882287510560859).
Actions: walk, run, squat, jump, stop + wave to camera; blended transitions and turns (blender/render_batch2.py).
Same palette/head-scale characters and character-level split as make_blender_jobs.py. Two held-out characters also get a
~19 s demo script with every action (split "demo": video only, never trained or gated).

  python character/make_blender_jobs2.py --asset <asset root> --out character/packs/human_v2 --chars 140 --heldout 25 --shards 4
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

REGIONS = {(255, 209, 159): ("skin", 40), (124, 88, 35): ("hair", 80), (115, 167, 196): ("shirt", 120), (21, 96, 189): ("pants", 160), (41, 41, 41): ("dark", 200)}
SEG = lambda a, n, f: {"action": a, "frames": int(n), "facing": f}


def clips_for(rng, scale=1.0):
    f = int(rng.choice([-1, 1])); g = lambda: int(rng.choice([-1, 1]))
    turn = bool(rng.random() < .5); n = lambda x: max(4, int(round(x * scale)))
    return [{"name": "walk", "segments": [SEG("Walk", n(24), f), SEG("Walk", n(24), -f if turn else f)], "phase0": float(rng.random())},
            {"name": "run", "segments": [SEG("Walk", n(12), g()), SEG("Run", n(36), "same")], "phase0": float(rng.random())},
            {"name": "squat", "segments": [SEG("Idle", n(8), g()), SEG("Squat", n(30), "same"), SEG("Idle", n(10), "same")], "phase0": float(rng.random())},
            {"name": "jump", "segments": [SEG("Idle", n(6), g()), SEG("Jump", n(26), "same"), SEG("Idle", n(16), "same")], "phase0": float(rng.random())},
            {"name": "wave", "segments": [SEG("Walk", n(16), g()), SEG("Idle", n(10), "same"), SEG("Idle", n(8), "camera"), SEG("Wave", n(30), "camera")],
             "phase0": float(rng.random())}]


def resolve_same(clips):
    for c in clips:
        last = 1
        for s in c["segments"]:
            if s["facing"] == "same": s["facing"] = last
            last = s["facing"] if s["facing"] != "camera" else last
    return clips


DEMO = [SEG("Walk", 96, 1), SEG("Run", 60, 1), SEG("Walk", 24, -1), SEG("Walk", 72, -1), SEG("Idle", 12, -1), SEG("Idle", 10, "camera"),
        SEG("Wave", 48, "camera"), SEG("Idle", 12, 1), SEG("Squat", 36, 1), SEG("Idle", 8, 1), SEG("Jump", 26, 1), SEG("Idle", 12, 1), SEG("Walk", 48, 1)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--asset", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chars", type=int, default=140); ap.add_argument("--heldout", type=int, default=25); ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261017); ap.add_argument("--abs-root", default=None); ap.add_argument("--demo", type=int, default=2)
    ap.add_argument("--clip-scale", type=float, default=1.0)
    ap.add_argument("--shading", default="cel", choices=["cel", "pbr"]); ap.add_argument("--samples", type=int, default=8); a = ap.parse_args(); rng = np.random.default_rng(a.seed); root = Path(a.asset)
    base = np.asarray(Image.open(root / "Blend/Textures/ClothedLightSkin.png").convert("RGB")); a.out.mkdir(parents=True, exist_ok=False)
    tex = a.out / "_textures"; tex.mkdir(); R = lambda p: str(Path(a.abs_root) / p.relative_to(a.out)) if a.abs_root else str(p.resolve())
    lab = np.zeros_like(base)
    for c, (_, g) in REGIONS.items(): lab[(base == c).all(-1)] = g
    Image.fromarray(lab).save(tex / "label.png"); chars, idx = [], []
    for k in range(a.chars):
        name = f"h{k:04d}"; split = "heldout" if k >= a.chars - a.heldout else "train"
        col = {"skin": (np.array([255, 212, 170]) * rng.uniform(.38, 1.0)).clip(0, 255), "hair": rng.integers(10, 170, 3), "shirt": rng.integers(25, 245, 3),
               "pants": rng.integers(20, 200, 3), "dark": rng.integers(10, 90, 3)}
        t = base.copy()
        for c, (rn, _) in REGIONS.items(): t[(base == c).all(-1)] = col[rn]
        Image.fromarray(t.astype(np.uint8)).save(tex / f"{name}.png")
        cl = resolve_same(clips_for(rng, a.clip_scale))
        if split == "heldout" and k < a.chars - a.heldout + a.demo:
            cl.append({"name": "demo", "segments": json.loads(json.dumps(DEMO)), "phase0": 0.0, "split": "demo"})
        chars.append({"name": name, "split": split, "texture": R(tex / f"{name}.png"), "label_texture": R(tex / "label.png"), "scale": 2.39,
                      "palette": {k_: [int(v) for v in v_] for k_, v_ in col.items()}, "head_scale": float(rng.uniform(.9, 1.15)), "clips": cl})
        for c in cl:
            idx.append({"clip": f"{name}/{c['name']}", "character": name, "action": c["name"], "split": c.get("split", split),
                        "frames": sum(s["frames"] for s in c["segments"])})
    for s in range(a.shards):
        (a.out / f"_batch_{s}.json").write_text(json.dumps({"fbx": "__FBX__", "out": R(a.out), "size": 256, "right_yaw": 3.14159265, "samples": a.samples, **({"shading": "pbr"} if a.shading == "pbr" else {}), "wave_bvh": "__WAVE__", "wave_extra_yaw": 3.14159265,
                                                             "characters": chars[s::a.shards]}, indent=1))
    (a.out / "index.json").write_text(json.dumps({"schema": "ff_cartoon_blender_v2", "split_by": "character", "seed": a.seed,
                                                  "regions": {v[0]: v[1] for v in REGIONS.values()},
                                                  "motion_sources": {"Walk/Run/Idle/Jump": "Quaternius Animated Human (CC0 1.0), asset keyframes",
                                                                     "Wave": "CMU mocap 141_16 'Wave Hello' (cgspeed BVH; CMU: no restrictions), retargeted by blender/retarget.py (Payton 1560886201718931488)",
                                                                     "Squat": "hand-keyed in blender/actions.py (procedural, on Idle)", "transitions": "8-frame pose cross-fades, 8-frame yaw easing"},
                                                  "clips": idx}, indent=1), newline="\n")
    print(json.dumps({"characters": a.chars, "clips": len(idx), "frames": sum(c["frames"] for c in idx if c["split"] != "demo"), "shards": a.shards}))


if __name__ == "__main__":
    main()

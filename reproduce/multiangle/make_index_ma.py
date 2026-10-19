"""GEN60 stage 1 (handoffs/scene_entity_texture/gen60/GEN60_CARD.md): index.json for the multi-angle pack so char_render.py
can train on it. Splits:
  train     = train characters, train clips (walk_r*, turn_r*, idle_r*)
  evalangle = train characters, eval_* clips (held-out yaw bands; never trained)
  heldout   = the held-out characters, all clips;  calib = the calib characters, all clips
Asserts: every clip dir has gt.json + all four passes complete; no 'train' frame yaw inside a held-out band (MANIFEST).
  python character/make_index_ma.py --pack character/packs/human_v5_multiangle [--allow-partial]"""
import argparse, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def in_bands(deg, bands):
    d = deg % 360
    return any(lo <= d <= hi or lo <= d + 360 <= hi for lo, hi in bands)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--allow-partial", action="store_true"); a = ap.parse_args()
    pk = Path(a.pack); man = json.loads((pk / "MANIFEST.json").read_text()); v4 = json.loads((HERE / "packs" / "human_v4" / "index.json").read_text())
    role = {c: s for s, cs in man["split"].items() for c in cs}
    bands = [b for bs in man["held_out"].values() for b in bs]
    clips, missing, bad_yaw = [], [], []
    for ch in sorted(role):
        cdir = pk / ch
        if not cdir.is_dir(): missing.append(ch); continue
        for c in sorted(p for p in cdir.iterdir() if p.is_dir()):
            gt = c / "gt.json"
            if not gt.exists(): missing.append(f"{ch}/{c.name}"); continue
            fr = json.loads(gt.read_text())["frames"]; n = len(fr)
            if any(len(list((c / ps).glob("*.png"))) != n for ps in ("rgba", "labels", "surf", "part")): missing.append(f"{ch}/{c.name} passes"); continue
            ev = c.name.startswith("eval_")
            split = role[ch] if role[ch] != "train" else ("evalangle" if ev else "train")
            if split == "train":
                bad_yaw += [f"{ch}/{c.name}:{f['t']}" for f in fr if in_bands(math.degrees(float(f["yaw"])) - 180.0, bands)]   # card angle = yaw - 180 deg (walk_r0 337.55 -> 157.55)
            clips.append({"clip": f"{ch}/{c.name}", "character": ch, "action": c.name, "split": split, "frames": n})
    out = {"schema": "ff_cartoon_blender_v2", "split_by": "character + yaw band", "source_manifest": "MANIFEST.json", "regions": v4["regions"],
           "angle_convention": man["angle_convention"], "held_out": man["held_out"], "clips": clips,
           "counts": {s: sum(1 for c in clips if c["split"] == s) for s in ("train", "evalangle", "heldout", "calib")},
           "frames": {s: sum(c["frames"] for c in clips if c["split"] == s) for s in ("train", "evalangle", "heldout", "calib")}}
    print(json.dumps({"counts": out["counts"], "frames": out["frames"], "missing": len(missing), "missing_examples": missing[:5], "bad_yaw": len(bad_yaw)}))
    if missing and not a.allow_partial: raise SystemExit("incomplete pack: %d missing" % len(missing))
    if bad_yaw: raise SystemExit("train frame inside a held-out band: %s" % bad_yaw[:5])
    (pk / "index.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

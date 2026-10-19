"""MULTIANGLE + PATTERN-OBJECT pack batches, v2 contract (Payton 1561517739947073642; Astra 1561517804467785762).
Same 140 human_v4 cartoon characters, textures, palettes, proportions and procedural shirt PATTERN parameters.
View angle a (degrees): 0 = profile facing right, 90 = back to camera, 180 = facing left, 270 = facing camera (front).
HELD-OUT angle bands, chosen on purpose, never in any training clip:
  INTERP  45 +-10 and 225 +-10   (fill in between seen angles)
  EXTRAP  270 +-25               (front view never seen at all)
TRAINING clips per character (continuous random yaws from the allowed region, seeded per character):
  6 x walk at a fixed random yaw (48 f); 2 x walk with a mid-clip TURN, a continuous linear sweep of >= 60 deg that stays
  inside one allowed arc (72 f: 12 f straight, 48 f turning, 12 f straight); 4 x idle at random yaws (4 f) = references.
  Allowed arcs: A1 = 55..215 (side through back to side), A3 = 295..395 (= 295..35, through the profile).
EVAL clips per character (tagged eval; never trained on):
  walk at exactly 45, 225 (INTERP), 270 (EXTRAP) and 90 (REAR: the shirt-pattern back-side truth check), 48 f each;
  a turn sweeping 200 -> 340 through the unseen front (72 f).
Characters keep their human_v4 split (train / heldout / calib). surface_pass on: surf + part passes, surface samples and
shirt track per frame, so the PATTERN OBJECT (child of the shirt, shirt-local surface coordinates, params = family,
colours, scale, phase, seed) can be checked against truth at every angle, including the rear.
  python forward_flow_cartoon/character/build_multiangle_batch.py [--pilot h0040] [--batches 8]"""
import argparse, glob, hashlib, json, os, random
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "packs", "human_v4")
INTERP = [(35, 55), (215, 235)]; EXTRAP = [(245, 295)]; ARCS = [(55, 215), (295, 395)]
WALK_N, TURN_N, IDLE_N = 48, 72, 4


def held(a):
    a %= 360
    return any(lo <= a <= hi for lo, hi in INTERP + EXTRAP)


def sample_allowed(rng):
    lo, hi = rng.choices(ARCS, weights=[b - a for a, b in ARCS])[0]
    return round(rng.uniform(lo + 1, hi - 1) % 360, 2)


def sample_turn(rng):
    lo, hi = rng.choices(ARCS, weights=[b - a for a, b in ARCS])[0]; span = rng.uniform(60, hi - lo - 2)
    a0 = rng.uniform(lo + 1, hi - 1 - span); a1 = a0 + span
    if rng.random() < 0.5: a0, a1 = a1, a0
    return round(a0, 2), round(a1, 2)


def clips_for(c, walk0):
    rng = random.Random(int(hashlib.md5(c["name"].encode()).hexdigest()[:8], 16)); out = []
    for i in range(6):
        a = sample_allowed(rng); out.append({"name": f"walk_r{i}", "role": "train", "segments": [{"action": "Walk", "frames": WALK_N, "facing": {"yaw_deg": a}}], "phase0": walk0})
    for i in range(2):
        a0, a1 = sample_turn(rng)
        out.append({"name": f"turn_r{i}", "role": "train", "phase0": walk0, "segments": [
            {"action": "Walk", "frames": 12, "facing": {"yaw_deg": a0}}, {"action": "Walk", "frames": 48, "facing": {"yaw_from": a0, "yaw_to": a1}},
            {"action": "Walk", "frames": 12, "facing": {"yaw_deg": a1}}]})
    for i in range(4):
        a = sample_allowed(rng); out.append({"name": f"idle_r{i}", "role": "train", "segments": [{"action": "Idle", "frames": IDLE_N, "facing": {"yaw_deg": a}}], "phase0": 0.0})
    for a, tag in ((45, "interp"), (225, "interp"), (270, "extrap_front"), (90, "rear")):
        out.append({"name": f"eval_{tag}_a{a:03d}", "role": "eval", "segments": [{"action": "Walk", "frames": WALK_N, "facing": {"yaw_deg": a}}], "phase0": walk0})
    out.append({"name": "eval_turn_front", "role": "eval", "phase0": walk0, "segments": [
        {"action": "Walk", "frames": 12, "facing": {"yaw_deg": 200}}, {"action": "Walk", "frames": 48, "facing": {"yaw_from": 200, "yaw_to": 340}},
        {"action": "Walk", "frames": 12, "facing": {"yaw_deg": 340}}]})
    for cl in out:                                                                 # contract check: no held-out angle in any training clip
        if cl["role"] == "train":
            for sg in cl["segments"]:
                f = sg["facing"]; angs = [f["yaw_deg"]] if "yaw_deg" in f else [f["yaw_from"] + (f["yaw_to"] - f["yaw_from"]) * k / 200 for k in range(201)]
                assert not any(held(x) for x in angs), (c["name"], cl["name"], f)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", default=""); ap.add_argument("--batches", type=int, default=8); ap.add_argument("--out", default="human_v5_multiangle"); a = ap.parse_args()
    chars, base = [], None
    for f in sorted(glob.glob(os.path.join(SRC, "_batch_*.json"))):
        b = json.load(open(f)); base = base or {k: v for k, v in b.items() if k != "characters"}; chars += b["characters"]
    if a.pilot: chars = [c for c in chars if c["name"] == a.pilot]; a.out += "_pilot"; a.batches = 1
    cfg = {k: v for k, v in base.items() if k not in ("wave_bvh", "wave_extra_yaw", "wave_plant")}
    cfg.update(out="/workspace/forward_flow_cartoon/character/packs/" + a.out, patterns=True, surface_pass=True, track_mode="face", timing=True)
    new = []
    for c in chars:
        walk0 = next((cl.get("phase0", 0.0) for cl in c["clips"] if cl["name"] == "walk"), 0.0); c2 = {k: v for k, v in c.items() if k != "clips"}
        c2["clips"] = clips_for(c, walk0)
        c2["pattern_object"] = {"id": f"{c['name']}/shirt/pattern", "parent": f"{c['name']}/shirt", "frame": "shirt-local surface coordinates (Generated)", "params": c.get("shirt")}
        new.append(c2)
    d = os.path.join(HERE, "packs", a.out); os.makedirs(d, exist_ok=True); n = max(1, a.batches)
    for i in range(n):
        part = new[i::n]
        if part: json.dump({**cfg, "characters": part}, open(os.path.join(d, f"_batch_{i}.json"), "w"), indent=1)
    fr = lambda role: sum(sg["frames"] for c in new for cl in c["clips"] if cl["role"] == role for sg in cl["segments"])
    man = {"contract": "Payton 1561517739947073642", "angle_convention": "0 profile facing right, 90 back, 180 facing left, 270 front", "held_out": {"interp": INTERP, "extrap_front": EXTRAP},
           "allowed_arcs": ARCS, "characters": len(new), "split": {s: sorted(c["name"] for c in new if c.get("split") == s) for s in sorted({c.get("split") for c in new})},
           "frames_train": fr("train"), "frames_eval": fr("eval"), "passes_per_frame": ["rgba", "labels", "surf", "part"],
           "pattern_families": sorted({(c.get("shirt") or {}).get("family", "none") for c in new}), "batches": n}
    json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"), indent=1); print(json.dumps({k: v for k, v in man.items() if k != "split"}), {k: len(v) for k, v in man["split"].items()})


if __name__ == "__main__":
    main()

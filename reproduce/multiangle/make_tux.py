"""GEN60 stage 2 (handoffs/scene_entity_texture/gen60/GEN60_CARD.md): the TUXEDO hold-out character. Same Quaternius mesh and the
existing shirt machinery (palette atlas + the "logo" patch), so render_batch is unchanged: black jacket (shirt region) with a
white shirt-front panel (logo patch), black trousers, black shoes. That outfit (colour layout) never occurs in the training pack.
No lapels / bow tie / tails (same mesh), disclosed.
  python character/make_tux.py --out character/packs/gen60_tux [--panels]   (--panels: 4 candidate panel placements, preview)"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import make_blender_jobs2 as M

HERE = Path(__file__).resolve().parent
POD = "/workspace/forward_flow_cartoon"
JACKET, WHITE, TROUSERS, SHOES = [2, 2, 3], [218, 218, 212], [24, 24, 28], [12, 12, 14]   # shirt pattern colours are LINEAR in the shader (as in training): 2 ~ sRGB 22
SKIN, HAIR = [214, 170, 132], [46, 30, 20]
PANELS = {"p0": [1.12, 1.20, 0.60, 0.76], "p1": [1.15, 1.23, 0.60, 0.76], "p2": [1.08, 1.18, 0.60, 0.76], "p3": [1.13, 1.19, 0.58, 0.78]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--panels", action="store_true"); ap.add_argument("--panel", default="p0"); a = ap.parse_args()
    out = Path(a.out); tex = out / "_textures"; tex.mkdir(parents=True, exist_ok=True)
    src = json.loads((HERE / "packs" / "human_v5_multiangle" / "_batch_0.json").read_text())        # same config as the pack
    base = np.asarray(Image.open(HERE.parent / "assets_cc0/human/Animated Human by @Quaternius/Blend/Textures/ClothedLightSkin.png").convert("RGB"))
    col = {"skin": SKIN, "hair": HAIR, "shirt": JACKET, "pants": TROUSERS, "dark": SHOES}
    t = base.copy()
    for c, (rn, _) in M.REGIONS.items(): t[(base == c).all(-1)] = col[rn]
    Image.fromarray(t.astype(np.uint8)).save(tex / "tux.png")
    lab = np.zeros(base.shape[:2], np.uint8)
    for c, (_, g) in M.REGIONS.items(): lab[(base == c).all(-1)] = g
    Image.fromarray(lab).save(tex / "label.png")
    proto = src["characters"][0]
    yaws = [0, 45, 90, 180, 225, 270, 315]
    clips = [{"name": f"view_a{y:03d}", "segments": [{"action": "Walk", "frames": 24, "facing": {"yaw_deg": y}}], "phase0": 0.0} for y in yaws]
    clips.append({"name": "turn_0_360", "segments": [{"action": "Walk", "frames": 72, "facing": {"yaw_from": 0, "yaw_to": 360}}], "phase0": 0.0})
    names = list(PANELS) if a.panels else [a.panel]
    chars = []
    for p in names:
        chars.append({**{k: v for k, v in proto.items() if k not in ("clips", "shirt", "pattern_object", "texture", "label_texture", "palette", "name", "split")},
                      "name": f"tux_{p}", "split": "gen60_holdout", "texture": f"{POD}/character/packs/{out.name}/_textures/tux.png",
                      "label_texture": f"{POD}/character/packs/{out.name}/_textures/label.png", "palette": col,
                      "shirt": {"family": "logo", "colors": [JACKET, WHITE], "scale": 8.0, "phase": 0.0, "seed": 0, "logo": PANELS[p], "red_present": False,
                                "note": "GEN60 tuxedo: black jacket + white shirt-front panel; outfit never in training"},
                      "clips": clips if (not a.panels or p == "p0") else clips[:1] + [clips[5]]})
    b = {**{k: v for k, v in src.items() if k != "characters"}, "out": f"{POD}/character/packs/{out.name}", "characters": chars}
    (out / "_batch_tux.json").write_text(json.dumps(b, indent=1), newline="\n")
    print(json.dumps({"characters": [c["name"] for c in chars], "clips": [len(c["clips"]) for c in chars], "proto_keys": sorted(proto)}))


if __name__ == "__main__":
    main()

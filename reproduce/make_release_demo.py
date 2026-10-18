"""Release action demo pack for crender_h3y (Payton 1561216783745032254: render walking, running, jumping, turning around,
waving, with the wave's floating legs fixed; RELEASE_RENDER_CARD.md).
Rebuilds the human_v3 character list with make_blender_jobs2.py's exact seed and arguments, so the held-out characters
(h0115+) are the same palettes the gates used, then keeps only N held-out characters with ONE 'demo' clip: the RELEASE
script below. wave_plant=true holds the hips and legs at Idle during the CMU wave (feet planted). Never trained or gated
on; video_demo.py renders truth | network and the wave foot-drift gate.
  python character/make_release_demo.py --asset <Quaternius root> --out character/packs/release_v1 --demo 3 --abs-root <absolute path of the pack>
"""
import argparse, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import make_blender_jobs2 as M

SEG = M.SEG                                                                      # 24 fps; facing 1 = right, -1 = left, "camera" = front
RELEASE = [SEG("Walk", 72, 1), SEG("Run", 72, 1), SEG("Walk", 24, -1), SEG("Walk", 48, -1),     # walk, run, turn around, walk back
           SEG("Idle", 12, -1), SEG("Jump", 26, -1), SEG("Idle", 12, -1),                         # jump
           SEG("Idle", 10, "camera"), SEG("Wave", 72, "camera"), SEG("Idle", 12, "camera"),        # turn to camera, wave (planted)
           SEG("Walk", 24, 1), SEG("Run", 48, 1), SEG("Idle", 12, 1), SEG("Jump", 26, 1), SEG("Idle", 16, 1)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--asset", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--demo", type=int, default=3); ap.add_argument("--abs-root", default=None); a = ap.parse_args()
    M.DEMO = RELEASE                                                             # same rng stream: palettes identical to human_v3
    sys.argv = ["make_blender_jobs2.py", "--asset", a.asset, "--out", str(a.out), "--chars", "140", "--heldout", "25", "--shards", "1",
                "--demo", str(a.demo)] + (["--abs-root", a.abs_root] if a.abs_root else [])
    M.main()
    b = json.loads((a.out / "_batch_0.json").read_text())
    keep = []
    for ch in b["characters"]:
        d = [c for c in ch["clips"] if c.get("split") == "demo"]
        if d: ch["clips"] = d; keep.append(ch)
    b["characters"] = keep; b["wave_plant"] = True
    (a.out / "_batch_0.json").write_text(json.dumps(b, indent=1))
    ix = json.loads((a.out / "index.json").read_text()); ix["clips"] = [c for c in ix["clips"] if c["split"] == "demo"]
    ix["release_script"] = RELEASE; ix["wave_plant"] = "hips + legs held at Idle frame 0 during the CMU 141_16 wave"
    (a.out / "index.json").write_text(json.dumps(ix, indent=1), newline="\n")
    names = {c["name"] for c in keep}
    for p in (a.out / "_textures").glob("h*.png"):
        if p.stem not in names: p.unlink()
    print(json.dumps({"characters": sorted(names), "frames_per_clip": sum(s["frames"] for s in RELEASE), "seconds": round(sum(s["frames"] for s in RELEASE) / 24, 1)}))


if __name__ == "__main__":
    main()

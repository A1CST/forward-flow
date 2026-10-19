"""GEN60 stage 3/4 compositor: street v4 (procedural scene renderer) + the character sprite of each frame along path60.json.
  python render/gen60_compose.py --sprites <dir of 00000.png ...> --out <dir> [--label TRUTH|MODEL] [--every 1]
Writes frames/, gen60.mp4, scene_log.json (per frame: drawn object ids, scene ms, compose ms) and a re-render check of 3 frames."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_scene_v4 as V4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sprites", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--path", default=str(Path(__file__).resolve().parent / "gen60" / "path60.json")); ap.add_argument("--label", default="")
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--gt", default="", help="gt.json: place the sprite canvas from the rig joints"); ap.add_argument("--seed", type=int, default=20261018); a = ap.parse_args()
    out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True); path = json.loads(Path(a.path).read_text())
    world = V4.make_street(a.seed); log = []; sp = Path(a.sprites)
    JS = [f["joints_px"] for f in json.loads(Path(a.gt).read_text())["frames"]] if a.gt else None
    for f in path["frames"][::a.every]:
        t0 = time.perf_counter(); spr = Image.open(sp / f"{f['f']:05d}.png").convert("RGBA")
        cam = V4.Cam(tuple(f["C"]), tuple(f["L"]), V4.F_PX * f.get("zoom", 1.0)); img, drawn = V4.render(world, cam, spr, tuple(f["P"]), joints=JS[f["f"]] if JS else None); t1 = time.perf_counter()
        if a.label: ImageDraw.Draw(img).text((8, 6), f"{a.label}  t={f['t']:5.2f}s  view={f['view_deg']:.0f}deg", fill=(20, 20, 20))
        img.save(out / "frames" / f"{f['f']:05d}.png"); log.append({"f": f["f"], "drawn": drawn, "scene_compose_ms": round(1000 * (t1 - t0), 2)})
    # determinism: re-render 3 frames from scratch, compare bytes of the decoded image
    chk = []
    for f in (path["frames"][0], path["frames"][len(path["frames"]) // 2], path["frames"][-1]):
        w2 = V4.make_street(a.seed); spr = Image.open(sp / f"{f['f']:05d}.png").convert("RGBA")
        im2, _ = V4.render(w2, V4.Cam(tuple(f["C"]), tuple(f["L"]), V4.F_PX * f.get("zoom", 1.0)), spr, tuple(f["P"]), joints=JS[f["f"]] if JS else None)
        if a.label: ImageDraw.Draw(im2).text((8, 6), f"{a.label}  t={f['t']:5.2f}s  view={f['view_deg']:.0f}deg", fill=(20, 20, 20))
        chk.append(hashlib.md5(im2.tobytes()).hexdigest() == hashlib.md5(Image.open(out / "frames" / f"{f['f']:05d}.png").convert("RGB").tobytes()).hexdigest())
    ms = sorted(x["scene_compose_ms"] for x in log)
    summary = {"frames": len(log), "buildings": len(world), "scene_compose_ms_median": ms[len(ms) // 2], "scene_compose_ms_p95": ms[int(len(ms) * .95)],
               "rerender_identical": chk, "seed": a.seed, "path": a.path}
    (out / "scene_log.json").write_text(json.dumps({"summary": summary, "frames": log}))
    fps = path["fps"] // a.every
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(out / "frames" / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(out / "gen60.mp4")], check=True)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

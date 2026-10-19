"""GEN60 demo video (Payton 1561563429037285580): reference card -> split screen (Blender truth left, network right, same
minute, same camera) with a live input panel (skeleton + camera / view angle fed to the model) -> pauses on the front view, the
face zoom and a camera move (truth and model side by side) -> numbers + GitHub link.
  python render/gen60_demo.py --out render/gen60/demo"""
import argparse, json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent; C = R.parent / "character"
W, H, FPS = 1280, 720, 24
BG, FG, DIM, ACC = (24, 26, 32), (236, 238, 242), (150, 156, 168), (240, 196, 80)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (8, 11), (11, 12)]


def font(sz, bold=False):
    for f in (("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"), "DejaVuSans.ttf"):
        try: return ImageFont.truetype(f, sz)
        except OSError: pass
    return ImageFont.load_default()


F = {k: font(*v) for k, v in {"h1": (40, True), "h2": (26, True), "b": (20, False), "s": (16, False), "sb": (16, True)}.items()}


def on_grey(rgba, size, grey=(205, 207, 212)):
    b = Image.new("RGBA", rgba.size, grey + (255,)); b.alpha_composite(rgba); return b.convert("RGB").resize(size, Image.NEAREST)


def ctext(d, y, s, f, fill=FG): d.text(((W - d.textlength(s, font=f)) / 2, y), s, font=f, fill=fill)


def intro(ref):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    ctext(d, 60, "The only image of this outfit the model saw", F["h1"])
    im.paste(on_grey(ref, (384, 384)), ((W - 384) // 2, 150))
    ctext(d, 560, "Tuxedo-style outfit: never in the training set (140 other characters, other outfits).", F["b"], DIM)
    ctext(d, 592, "Next: one continuous minute, Blender truth on the left, the network on the right, same camera.", F["b"], DIM)
    return im


def panel(J, f):
    p = Image.new("RGB", (140, 214), (14, 15, 19)); d = ImageDraw.Draw(p); s = 110 / 256
    for a, b in BONES: d.line([(15 + J[a][0] * s, 22 + J[a][1] * s), (15 + J[b][0] * s, 22 + J[b][1] * s)], fill=ACC, width=2)
    for x, y in J: d.ellipse([15 + x * s - 2, 22 + y * s - 2, 15 + x * s + 2, 22 + y * s + 2], fill=FG)
    d.text((6, 140), f"view {f['view_deg']:.0f} deg", font=F["s"], fill=FG); d.text((6, 160), f"zoom {f.get('zoom', 1):.1f}x", font=F["s"], fill=FG)
    d.text((6, 180), f"cam az {f['azimuth_deg']:.0f} deg", font=F["s"], fill=FG)
    d.text((8, 2), "MODEL INPUT", font=F["sb"], fill=DIM); return p


def split(tr, mo, J, f, banner=None):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((20, 18), "BLENDER TRUTH (target)", font=F["h2"], fill=FG); d.text((580, 18), "NETWORK (drawn from controls)", font=F["h2"], fill=FG)
    im.paste(tr.resize((540, 540)), (20, 60)); im.paste(mo.resize((540, 540)), (580, 60)); im.paste(panel(J, f), (1134, 386))
    d.text((20, 620), f"t = {f['t']:5.2f} s", font=F["b"], fill=FG)
    d.rectangle([160, 630, 1120, 638], fill=(60, 64, 74)); d.rectangle([160, 630, 160 + 960 * f["f"] / 1439, 638], fill=ACC)
    if banner:
        d.rectangle([0, 300, W, 360], fill=(0, 0, 0)); ctext(d, 312, banner, F["h2"], ACC)
    return im


def pause(tr_spr, mo_spr, f, title):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); ctext(d, 30, f"PAUSE  {title}", F["h1"], ACC)
    ctext(d, 84, f"t = {f['t']:.1f} s   view {f['view_deg']:.0f} deg   (left: Blender truth, right: network; character only, 2x)", F["b"], DIM)
    for i, (spr, lab) in enumerate(((tr_spr, "TRUTH"), (mo_spr, "NETWORK"))):
        x = 140 + i * 520; im.paste(on_grey(spr, (480, 480)), (x, 130)); d.text((x, 618), lab, font=F["h2"], fill=FG)
    return im


def endcard(nums):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); ctext(d, 40, "The numbers", F["h1"])
    y = 120
    for k, v in nums:
        d.text((150, y), k, font=F["b"], fill=DIM); d.text((620, y), v, font=F["b"], fill=FG); y += 44
    ctext(d, 620, "github.com/A1CST/forward-flow", F["h2"], ACC)
    ctext(d, 664, "Character renderer only: stateless, one frame from (reference, 13 joints, view angle). Pose comes from the rig.", F["s"], DIM)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(R / "gen60" / "demo")); a = ap.parse_args()
    out = Path(a.out); fr_dir = out / "frames"; fr_dir.mkdir(parents=True, exist_ok=True)
    path = json.loads((R / "gen60" / "path60.json").read_text())["frames"]; gt = json.loads((C / "packs/gen60_truth_v2_ol/tux/gen60/gt.json").read_text())["frames"]
    ref = Image.open(C / "packs/gen60_truth_v2_ol/tux/ref.png").convert("RGBA"); drift = json.loads((R / "gen60" / "drift.json").read_text())
    sc = json.loads((R / "gen60/gen2/h5ma/score.json").read_text()); scc = json.loads((R / "gen60/gen2/h4p/score.json").read_text())
    TR, MO = R / "gen60/truth_v3/frames", R / "gen60/model_v2/frames"
    TS, MS = C / "packs/gen60_truth_v2_ol/tux/gen60/rgba", R / "gen60/gen2/h5ma/sprites"
    PAUSES = {456: "front view (an angle the model never trained on)", 552: "face zoom", 1104: "camera move (orbit, front-quarter)"}
    n = 0
    def put(im, k=1):
        nonlocal n
        for _ in range(k): im.save(fr_dir / f"{n:05d}.png"); n += 1
    put(intro(ref), 4 * FPS)
    for f in path:
        i = f["f"]; tr, mo = Image.open(TR / f"{i:05d}.png").convert("RGB"), Image.open(MO / f"{i:05d}.png").convert("RGB")
        put(split(tr, mo, gt[i]["joints_px"], f))
        if i in PAUSES:
            put(split(tr, mo, gt[i]["joints_px"], f, "PAUSE: " + PAUSES[i]), 12)
            put(pause(Image.open(TS / f"{i:05d}.png").convert("RGBA"), Image.open(MS / f"{i:05d}.png").convert("RGBA"), f, PAUSES[i]), int(2.5 * FPS))
    m, c = drift["model"], drift["control"]
    nums = [("Model size", "6.3 M parameters (25 MB fp32)"),
            ("Render speed (RTX 4090)", f"{sc['ms_per_frame_batched']:.2f} ms/frame batched, {sc['ms_per_frame_single']:.1f} ms single"),
            ("Street + composite (CPU)", "~13 ms/frame median"),
            ("Drift, colour error s1 -> s60", f"{m['err_s1']:.1f} -> {m['err_s60']:.1f}   (0-255, inside the true figure)"),
            ("Drift, outline IoU s1 -> s60", f"{m['iou_s1']:.3f} -> {m['iou_s60']:.3f}"),
            ("Whole minute vs side-trained model", f"colour err {m['err_mean']:.1f} vs {c['err_mean']:.1f};  IoU {m['iou_mean']:.3f} vs {c['iou_mean']:.3f}"),
            ("Misses", "white shirt front lost, jacket greyer than truth, soft face")]
    put(endcard(nums), 9 * FPS)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(fr_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(out / "gen60_demo.mp4")], check=True)
    print(json.dumps({"frames": n, "seconds": round(n / FPS, 1), "out": str(out / "gen60_demo.mp4")}))


if __name__ == "__main__":
    main()

"""Forward Flow character renderer: try it.

Draws a cartoon character in any pose from ONE reference image of that character plus a sequence of 13 2-D joints.
The network never sees target frames, only the reference image and the joints.

  python try_it.py --example h0115                         # held-out character, 20 s action script -> out/h0115.mp4
  python try_it.py --example h0115 --recolor 0.33          # same motion, reference image hue-rotated by 0.33 turns
  python try_it.py --ref my_ref.png --ref_joints my_ref.json --joints examples/h0116/joints.json --out out/mine.mp4

--ref must be the same kind of image as examples/*/ref.png: a 256x256 RGBA figure-alone cutout, idle pose, facing right,
with its 13 joints in --ref_joints (examples/*/ref.json, key joints_px). Output: frames over a scrolling street + an mp4
(needs ffmpeg on PATH; frames are written either way).
"""
import argparse, colorsys, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "forward_flow"))
import char_render as C
from street import street


def recolor(img, turns):
    """Hue-rotate the figure's colours (alpha kept) to make a 'new' character from an example one."""
    a = np.asarray(img.convert("RGBA"), np.float32) / 255; rgb = a[..., :3].reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in rgb]); hsv[:, 0] = (hsv[:, 0] + turns) % 1.0
    a[..., :3] = np.array([colorsys.hsv_to_rgb(*p) for p in hsv]).reshape(a.shape[:2] + (3,))
    return Image.fromarray((a * 255).round().astype(np.uint8), "RGBA")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--example", default=None); ap.add_argument("--ref"); ap.add_argument("--ref_joints")
    ap.add_argument("--joints"); ap.add_argument("--recolor", type=float, default=0.0); ap.add_argument("--model", default=str(HERE / "weights" / "crender_h3y"))
    ap.add_argument("--out", default=None); ap.add_argument("--no_street", action="store_true"); a = ap.parse_args()
    if a.example:
        ex = HERE / "examples" / a.example; a.ref = a.ref or str(ex / "ref.png"); a.ref_joints = a.ref_joints or str(ex / "ref.json"); a.joints = a.joints or str(ex / "joints.json")
    out = Path(a.out or f"out/{a.example or 'render'}{'_recolor' if a.recolor else ''}.mp4"); fd = out.with_suffix(""); fd.mkdir(parents=True, exist_ok=True)
    d = C.dev(); net = C.load_model(a.model).to(d)
    img = Image.open(a.ref).convert("RGBA"); img = recolor(img, a.recolor) if a.recolor else img; img.save(fd / "reference_used.png")
    ref = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)
    rj = torch.tensor(json.loads(Path(a.ref_joints).read_text())["joints_px"], dtype=torch.float32)
    z = net.tok(ref[None].to(d), rj[None].to(d) if net.v2 else None)
    fr = json.loads(Path(a.joints).read_text())["frames"]
    J = torch.tensor(np.asarray([f["joints_px"] for f in fr], np.float32), device=d)
    Y = torch.tensor([float(f.get("yaw", math.pi)) - math.pi for f in fr], device=d)
    P = torch.cat([net(J[i:i + 16], z=z.expand(len(J[i:i + 16]), -1), yaw=Y[i:i + 16]) for i in range(0, len(J), 16)]).cpu().numpy()
    Jn = J.cpu().numpy(); gpx = float(np.percentile(np.maximum(Jn[:, 10, 1], Jn[:, 12, 1]), 90)) + 3; cam = 0.0
    for t in range(len(P)):
        if t:
            ank = (10, 12)[int(np.argmax([Jn[t, 10, 1], Jn[t, 12, 1]]))]; cam -= Jn[t, ank, 0] - Jn[t - 1, ank, 0]   # street scrolls with the stance foot
        bg = np.full((256, 256, 3), 245, np.uint8) if a.no_street else street(cam, 256, seed=5, ground_px=gpx)
        x = P[t]; al = x[3][..., None]
        frame = (x[:3].transpose(1, 2, 0) * 255 + bg.astype(np.float32) * (1 - al)).clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).resize((512, 512), Image.NEAREST).save(fd / f"{t:05d}.png")
    fps = json.loads(Path(a.joints).read_text()).get("fps", 24)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fd / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
        print("wrote", out)
    else:
        print("ffmpeg not found; frames in", fd)


if __name__ == "__main__":
    main()

"""Forward Flow toy demo: a random action script, rendered live by the network from ONE reference image.

What happens:
1. A random but bounded script is drawn from the supported actions (Walk, Run, Jump, Idle, Wave, turns), using
   examples/action_library.json: joint segments recorded from the Blender release script. Each segment is also available
   mirrored (x flipped, left/right joints swapped, facing reversed); a mirrored wave uses the other hand.
   Only segments whose facing_in matches the current facing can follow, and joints and yaw cross-fade over 6 frames
   at every cut.
2. The network (weights/crender_h3y) draws every frame from the reference image + that frame's 13 joints. This is
   NETWORK RENDERING only; there is no Blender truth in this demo.
3. Frames are shown live (OpenCV window) at the script's 24 fps when the network keeps up, and the measured render
   time is printed.

  python toy_demo.py                                  # random script, seed printed
  python toy_demo.py --seed 7 --seconds 30 --example h0116 --recolor 0.5
  python toy_demo.py --replay out/toy_seed7.script.json --save out/replay.mp4 --no_window
The generated script is always saved as out/toy_seed<seed>.script.json, so any run can be replayed exactly.
"""
import argparse, colorsys, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "forward_flow"))
import char_render as C
from street import street

S, FADE = 256, 6
SWAP = [0, 1, 5, 6, 7, 2, 3, 4, 8, 11, 12, 9, 10]                 # joint order: head neck r_sh r_el r_wr l_sh l_el l_wr pelvis r_kn r_an l_kn l_an


def mirror_facing(f): return f if f == "camera" else -f


def library():
    L = json.loads((HERE / "examples" / "action_library.json").read_text())["segments"]; out = []
    for s in L:
        J = np.asarray([f["joints_px"] for f in s["frames"]], np.float32); Y = np.asarray([f["yaw"] for f in s["frames"]], np.float32)
        out.append(dict(s, J=J, Y=Y, mirrored=False))
        Jm = J[:, SWAP].copy(); Jm[..., 0] = S - Jm[..., 0]
        out.append(dict(s, J=Jm, Y=(3 * math.pi - Y), facing_in=mirror_facing(s["facing_in"]), facing_out=mirror_facing(s["facing_out"]), mirrored=True))
    return out


def make_script(lib, seconds, rng):
    """Bounded random walk over segments with matching facing; avoids repeating the same action back to back."""
    facing, last, seq, n = 1, None, [], 0
    while n < seconds * 24:
        cand = [i for i, s in enumerate(lib) if s["facing_in"] == facing and s["action"] != last]
        if not cand: cand = [i for i, s in enumerate(lib) if s["facing_in"] == facing]
        i = int(rng.choice(cand)); s = lib[i]; seq.append(i); n += len(s["J"]); facing, last = s["facing_out"], s["action"]
    return seq


def stitch(lib, seq):
    J, Y, act = [], [], []
    for k, i in enumerate(seq):
        s = lib[i]; j, y = s["J"].copy(), s["Y"].copy()
        if k and J:
            pj, py = J[-1], Y[-1]
            for t in range(min(FADE, len(j))):                    # cross-fade joints and yaw (shortest angle) at the cut
                w = (t + 1) / (FADE + 1); j[t] = (1 - w) * pj + w * j[t]
                dy = math.atan2(math.sin(y[t] - py), math.cos(y[t] - py)); y[t] = py + w * dy
        J += list(j); Y += list(y); act += [s["action"] + (" (mirrored)" if s["mirrored"] else "")] * len(j)
    return np.asarray(J, np.float32), np.asarray(Y, np.float32), act


def recolor(img, turns):
    a = np.asarray(img.convert("RGBA"), np.float32) / 255; rgb = a[..., :3].reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in rgb]); hsv[:, 0] = (hsv[:, 0] + turns) % 1.0
    a[..., :3] = np.array([colorsys.hsv_to_rgb(*p) for p in hsv]).reshape(a.shape[:2] + (3,))
    return Image.fromarray((a * 255).round().astype(np.uint8), "RGBA")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--example", default="h0115"); ap.add_argument("--recolor", type=float, default=0.0); ap.add_argument("--replay", default=None)
    ap.add_argument("--save", default=None); ap.add_argument("--no_window", action="store_true"); ap.add_argument("--model", default=str(HERE / "weights" / "crender_h3y"))
    a = ap.parse_args(); lib = library(); Path("out").mkdir(exist_ok=True)
    if a.replay:
        sc = json.loads(Path(a.replay).read_text()); seq, a.seed = sc["segments"], sc["seed"]; a.example, a.recolor = sc["example"], sc["recolor"]
    else:
        a.seed = int(np.random.default_rng().integers(0, 10_000)) if a.seed is None else a.seed
        seq = make_script(lib, a.seconds, np.random.default_rng(a.seed))
    sp = Path(f"out/toy_seed{a.seed}.script.json")
    sp.write_text(json.dumps({"seed": a.seed, "example": a.example, "recolor": a.recolor, "segments": [int(i) for i in seq],
                              "actions": [lib[i]["action"] + (" (mirrored)" if lib[i]["mirrored"] else "") for i in seq]}, indent=1))
    print("seed", a.seed, "->", sp, "|", " > ".join(lib[i]["action"] for i in seq))
    J, Y, act = stitch(lib, seq)
    d = C.dev(); net = C.load_model(a.model).to(d); ex = HERE / "examples" / a.example
    img = Image.open(ex / "ref.png").convert("RGBA"); img = recolor(img, a.recolor) if a.recolor else img
    ref = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)
    rj = torch.tensor(json.loads((ex / "ref.json").read_text())["joints_px"], dtype=torch.float32)
    z = net.tok(ref[None].to(d), rj[None].to(d) if net.v2 else None)
    try:
        import cv2
    except ImportError:
        cv2 = None
    show = cv2 is not None and not a.no_window; fd = Path(a.save).with_suffix("") if a.save else None
    if fd: fd.mkdir(parents=True, exist_ok=True)
    gpx = float(np.percentile(np.maximum(J[:, 10, 1], J[:, 12, 1]), 90)) + 3; cam = 0.0; ms = []; t_start = time.perf_counter()
    for t in range(len(J)):
        q = time.perf_counter()
        x = net(torch.from_numpy(J[t:t + 1]).to(d), z=z, yaw=torch.tensor([float(Y[t]) - math.pi], device=d))[0]
        if d.type == "cuda": torch.cuda.synchronize()
        ms.append(1e3 * (time.perf_counter() - q)); x = x.cpu().numpy()
        if t:
            ank = (10, 12)[int(np.argmax([J[t, 10, 1], J[t, 12, 1]]))]; cam -= J[t, ank, 0] - J[t - 1, ank, 0]
        bg = street(cam, S, seed=5, ground_px=gpx); al = x[3][..., None]
        fr = (x[:3].transpose(1, 2, 0) * 255 + bg.astype(np.float32) * (1 - al)).clip(0, 255).astype(np.uint8)
        big = np.asarray(Image.fromarray(fr).resize((512, 512), Image.NEAREST))
        if fd: Image.fromarray(big).save(fd / f"{t:05d}.png")
        if show:
            v = cv2.cvtColor(big, cv2.COLOR_RGB2BGR); cv2.putText(v, f"network render | {act[t]} | {ms[-1]:.1f} ms", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
            cv2.imshow("Forward Flow toy demo (network rendering)", v)
            wait = max(1, int(1000 / 24 - (time.perf_counter() - q) * 1000))
            if cv2.waitKey(wait) & 0xFF in (27, ord("q")): break
    wall = time.perf_counter() - t_start
    print(f"frames {len(ms)} | network {np.mean(ms):.2f} ms/frame mean, {np.percentile(ms, 95):.2f} p95 ({1000 / np.mean(ms):.0f} fps) on {d} | wall {wall:.1f} s")
    if show: cv2.destroyAllWindows()
    if fd and shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(fd / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.save], check=True)
        print("wrote", a.save)


if __name__ == "__main__":
    main()

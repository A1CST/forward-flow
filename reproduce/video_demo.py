"""20 s "walking down a street" demo for a held-out character (Claude 2026-10-17; Payton 1560861550703546390).

The model gets ONE reference image of a character it never trained on, plus the joints of a 20 s Blender walk (with a turn).
Left: Blender truth. Right: the network's frames. Both over the same scrolling street; the street scrolls at the stance
foot's speed so the feet do not slide. Also writes identity-drift numbers: per-region colour error of the model vs truth in
each 2 s window (should stay flat if identity holds).

  python character/video_demo.py --pack character/packs/human_v1 --model character/runs/crender_h1 --out character/runs/crender_h1/demo
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE), str(HERE.parent / "forward_flow")]
import char_render as C, cartoon_world as CW
from street import street


def over(bg, x):
    x = x.cpu().numpy(); a = x[3][..., None]
    return (x[:3].transpose(1, 2, 0) * 255 + bg.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); d = C.dev(); net = C.load_model(a.model).to(d); root = Path(a.pack); out = Path(a.out); gn = torch.Generator().manual_seed(11)
    wave_gate = {}
    ix = json.loads((root / "index.json").read_text()); regions = {k: (v,) for k, v in ix.get("regions", {}).items()}; summary = {}
    for c in [c for c in ix["clips"] if c["split"] == "demo"]:
        gt = json.loads((root / c["clip"] / "gt.json").read_text()); ch = c["character"]; fd = out / ch / "frames"; fd.mkdir(parents=True, exist_ok=True)
        ref = C.load_rgba(root / ch / "ref.png"); rj = torch.tensor(json.loads((root / ch / "ref.json").read_text())["joints_px"], dtype=torch.float32)
        z = net.tok(ref[None].to(d), (rj + torch.randn(13, 2, generator=gn) * C.REF_JOINT_NOISE)[None].to(d) if net.v2 else None)
        J = torch.tensor(np.asarray([f["joints_px"] for f in gt["frames"]], np.float32), device=d)
        Yw = torch.tensor([float(f.get("yaw", math.pi)) - math.pi for f in gt["frames"]], device=d)          # declared facing (scene state)
        P = torch.cat([net(J[i:i + 16], z=z.expand(len(J[i:i + 16]), -1), yaw=Yw[i:i + 16]) for i in range(0, len(J), 16)])
        cam = 0.0; win = {}; Jn = J.cpu().numpy(); gpx = float(np.percentile(np.maximum(Jn[:, 10, 1], Jn[:, 12, 1]), 90)) + 3
        for t, f in enumerate(gt["frames"]):
            if t:
                lo = int(np.argmax([Jn[t, 10, 1], Jn[t, 12, 1]])); ank = (10, 12)[lo]
                cam -= (Jn[t, ank, 0] - Jn[t - 1, ank, 0])                     # stance foot speed (px) -> street scroll (no sliding)
            bg = street(cam, 256, seed=5, ground_px=gpx)
            T = C.premul(C.load_rgba(root / c["clip"] / "rgba" / f"{t:05d}.png")).to(d)
            lab = torch.from_numpy(np.asarray(Image.open(root / c["clip"] / "labels" / f"{t:05d}.png").convert("L")).astype(np.int16)).to(d)
            for rn, ids in regions.items():
                e = C.region_err(P[t], T, lab, ids)
                if e is not None: win.setdefault(t // 48, {}).setdefault(rn, []).append(e)
            row = np.concatenate([over(bg, T), np.full((256, 4, 3), 30, np.uint8), over(bg, P[t])], 1)
            img = Image.fromarray(np.concatenate([np.full((24, row.shape[1], 3), 22, np.uint8), row], 0)); dr = ImageDraw.Draw(img)
            dr.text((6, 6), f"held-out character {ch} (never trained on)   left: Blender truth   right: network, from ONE reference image   {f.get('action', '')}", fill=(235, 235, 235))
            img.resize((img.width * 2, img.height * 2), Image.NEAREST).save(fd / f"{t:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(fd / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", str(out / ch / "video.mp4")], check=True)
        summary[ch] = {f"{w * 2}-{w * 2 + 2}s": {rn: round(float(np.mean(v)), 2) for rn, v in r.items()} for w, r in sorted(win.items())}
        wv = [t for t, f in enumerate(gt["frames"]) if f.get("action") == "Wave" and f.get("blend_w", 1) >= 1]    # wave foot-drift gate
        if wv:
            ank = Jn[wv][:, [10, 12]]; tdr = float(np.abs(ank - ank[0]).max())
            fp = []
            for t in wv:
                al = P[t][3].cpu().numpy(); ys, xs = np.nonzero(al > .5)
                if len(ys) == 0: continue
                low = ys >= ys.max() - 12; fp.append([float(xs[low].mean()), float(ys.max())])
            fp = np.asarray(fp); md = np.abs(fp - fp[0]).max(1) if len(fp) else np.zeros(1)
            wave_gate[ch] = {"wave_frames": len(wv), "truth_ankle_drift_px_max": round(tdr, 3), "model_foot_drift_px_p95": round(float(np.percentile(md, 95)), 3),
                             "model_foot_drift_px_max": round(float(md.max()), 3), "pass": bool(tdr <= 1.0 and np.percentile(md, 95) <= 2.0)}
    (out / "drift.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))
    if wave_gate:                                                                       # bars frozen in RELEASE_RENDER_CARD.md
        (out / "wave_foot_gate.json").write_text(json.dumps({"bars": {"truth_ankle_drift_px_max": 1.0, "model_foot_drift_px_p95": 2.0}, "characters": wave_gate,
                                                            "all_pass": all(g["pass"] for g in wave_gate.values())}, indent=1)); print("WAVE_GATE", json.dumps(wave_gate))


if __name__ == "__main__":
    main()

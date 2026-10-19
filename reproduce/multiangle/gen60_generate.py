"""GEN60 stage 4 (handoffs/scene_entity_texture/gen60/GEN60_CARD.md): the character model draws the 60 s tuxedo shot.
Inputs per frame = the ONE tuxedo reference (ref.png + its joints, same 2 px declared reader noise and seed as char_render.gate),
the target 13 joints from the Blender rig (declared pose) and the view yaw. Outputs straight-RGBA sprites 00000.png ... and
scores against the Blender truth frames (alpha soft IoU, in-mask RGB error 0-255) per view band, plus the timing.
  python character/gen60_generate.py --truth character/packs/gen60_truth_v2/tux --model character/runs/crender_h5ma --out <dir>"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import char_render as CR
from gate_bands import band_of


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--truth", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="gen60"); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ref2", default="", help="second reference CHARACTER dir (ref.png + ref.json): outfit swap (Payton 1561558577972379658)")
    ap.add_argument("--swap_at", type=int, default=720); ap.add_argument("--blend", type=int, default=0, help="frames of linear token blend (0 = hard swap)")
    ap.add_argument("--refs", default="", help="comma list of character dirs: outfit schedule after the primary (Payton 1561558890834034689)")
    ap.add_argument("--swap_frames", default="", help="comma list of frames where the next outfit starts (one per --refs entry)"); a = ap.parse_args()
    d = CR.dev(); net = CR.load_model(a.model).to(d).eval(); tr = Path(a.truth); out = Path(a.out); (out / "sprites").mkdir(parents=True, exist_ok=True)
    gt = json.loads((tr / a.clip / "gt.json").read_text())["frames"]; refJ = torch.tensor(json.loads((tr / "ref.json").read_text())["joints_px"], dtype=torch.float32)
    gn = torch.Generator().manual_seed(11); rjn = (refJ + torch.randn(13, 2, generator=gn) * CR.REF_JOINT_NOISE)[None].to(d)
    ref = CR.load_rgba(tr / "ref.png")[None].to(d)
    J = torch.tensor(np.stack([np.asarray(f["joints_px"], np.float32) for f in gt]), device=d)
    Y = torch.tensor([float(f["yaw"]) - math.pi for f in gt], dtype=torch.float32, device=d)
    with torch.no_grad():
        z = net.tok(ref, rjn if net.v2 else None); Z = z.expand(len(J), -1).clone()
        if a.ref2:                                                   # per-frame token: reference 1 before swap_at, reference 2 after (optional blend)
            r2 = Path(a.ref2); rj2 = torch.tensor(json.loads((r2 / "ref.json").read_text())["joints_px"], dtype=torch.float32)
            z2 = net.tok(CR.load_rgba(r2 / "ref.png")[None].to(d), (rj2 + torch.randn(13, 2, generator=gn) * CR.REF_JOINT_NOISE)[None].to(d) if net.v2 else None)
            w = ((torch.arange(len(J), device=d).float() - a.swap_at + 1) / max(1, a.blend)).clamp(0, 1)[:, None] if a.blend else (torch.arange(len(J), device=d) >= a.swap_at).float()[:, None]
            Z = (1 - w) * z + w * z2
        if a.refs:                                                   # hard outfit schedule: primary until swap_frames[0], refs[0] until swap_frames[1], ...
            for rd, fr in zip(a.refs.split(","), [int(v) for v in a.swap_frames.split(",")]):
                r2 = Path(rd); rj2 = torch.tensor(json.loads((r2 / "ref.json").read_text())["joints_px"], dtype=torch.float32)
                z2 = net.tok(CR.load_rgba(r2 / "ref.png")[None].to(d), (rj2 + torch.randn(13, 2, generator=gn) * CR.REF_JOINT_NOISE)[None].to(d) if net.v2 else None)
                Z[fr:] = z2
        for _ in range(2): net(J[:a.bs], z=z.expand(min(a.bs, len(J)), -1), yaw=Y[:a.bs])          # warm-up
        torch.cuda.synchronize() if d.type == "cuda" else None; t0 = time.perf_counter(); outs = []
        for i in range(0, len(J), a.bs):
            n = min(a.bs, len(J) - i); outs.append(net(J[i:i + n], z=Z[i:i + n], yaw=Y[i:i + n]))
        torch.cuda.synchronize() if d.type == "cuda" else None; t_batch = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(48): net(J[i:i + 1], z=z, yaw=Y[i:i + 1])
        torch.cuda.synchronize() if d.type == "cuda" else None; t_single = (time.perf_counter() - t0) / 48
        P = torch.cat(outs)
    acc = {}
    for k, f in enumerate(gt):
        p = P[k]; a_ = p[3:4].clamp_min(1e-6); rgb = (p[:3] / a_).clamp(0, 1) * (p[3:4] > 1e-3)
        Image.fromarray((torch.cat([rgb, p[3:4]], 0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8), "RGBA").save(out / "sprites" / f"{f['t']:05d}.png")
        T = CR.premul(CR.load_rgba(tr / a.clip / "rgba" / f"{f['t']:05d}.png")).to(d)
        b = band_of(math.degrees(float(f["yaw"]) - math.pi)); s = acc.setdefault(b, {"iou": [], "rgb": []})
        s["iou"].append(CR.soft_iou(p[3], T[3])); s["rgb"].append(CR.rgb_err(p, T))
    allv = {"iou": sum((v["iou"] for v in acc.values()), []), "rgb": sum((v["rgb"] for v in acc.values()), [])}
    res = {"model": a.model, "reference": str(tr / "ref.png"), "ref2": a.ref2, "swap_at": a.swap_at if a.ref2 else None, "blend": a.blend if a.ref2 else None, "refs": a.refs, "swap_frames": a.swap_frames, "frames": len(gt), "device": str(d),
           "ms_per_frame_batched": round(1000 * t_batch / len(gt), 3), "ms_per_frame_single": round(1000 * t_single, 3), "batch": a.bs,
           "overall": {"alpha_soft_iou": round(float(np.mean(allv["iou"])), 4), "rgb_err_in_mask_0_255": round(float(np.mean(allv["rgb"])), 2)},
           "bands": {b: {"frames": len(v["iou"]), "alpha_soft_iou": round(float(np.mean(v["iou"])), 4), "rgb_err_in_mask_0_255": round(float(np.mean(v["rgb"])), 2)} for b, v in sorted(acc.items())}}
    (out / "score.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

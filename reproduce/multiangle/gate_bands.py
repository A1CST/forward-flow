"""GEN60 stage 1 gate (handoffs/scene_entity_texture/gen60/GEN60_CARD.md): char_render model quality PER VIEW-ANGLE BAND on the
multi-angle pack. Same metrics as char_render.gate (alpha soft IoU, in-mask RGB error 0-255, token-swap identity) but binned by
the card's angle (0 profile right, 90 back, 180 profile left, 270 front). Run once for the test model and once for the control.
  python character/gate_bands.py --pack character/packs/human_v5_multiangle --model <run> --out <json> [--splits evalangle heldout]"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
import char_render as CR

BANDS = {"side": [(-25, 25), (155, 205)], "interp_45": [(35, 55)], "interp_225": [(215, 235)], "rear_90": [(65, 115)], "front_270": [(245, 295)]}


def band_of(deg):
    d = deg % 360
    for name, bs in BANDS.items():
        for lo, hi in bs:
            if lo <= d <= hi or lo <= d - 360 <= hi: return name
    return "other"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["evalangle", "heldout"]); ap.add_argument("--stride", type=int, default=4); a = ap.parse_args()
    d = CR.dev(); net = CR.load_model(a.model).to(d).eval(); res = {"model": a.model, "pack": a.pack, "bands_deg": BANDS, "splits": {}}
    with torch.no_grad():
        for sp in a.splits:
            P = CR.Pack(a.pack, sp, stride=a.stride); gn = torch.Generator().manual_seed(11)
            rjn = {c: (P.ref_J[c] + torch.randn(13, 2, generator=gn) * CR.REF_JOINT_NOISE)[None].to(d) for c in P.chars}
            toks = {c: net.tok(P.refs[c][None].to(d), rjn[c] if net.v2 else None) for c in P.chars}
            acc = {}
            for i in range(0, len(P.items), 32):
                idx = list(range(i, min(i + 32, len(P.items)))); J, R, T, _ = P.batch(idx, d); Y = P.yaw(idx, d)
                own = [P.items[k][1] for k in idx]; oth = [P.chars[(P.chars.index(c) + 1) % len(P.chars)] for c in own]
                out = net(J, z=torch.cat([toks[c] for c in own]), yaw=Y); sw = net(J, z=torch.cat([toks[c] for c in oth]), yaw=Y)
                for j, k in enumerate(idx):
                    b = band_of(math.degrees(P.yaws[k])); s = acc.setdefault(b, {"iou": [], "rgb": [], "swap_rgb": []})
                    s["iou"].append(CR.soft_iou(out[j][3], T[j][3])); s["rgb"].append(CR.rgb_err(out[j], T[j])); s["swap_rgb"].append(CR.rgb_err(sw[j], T[j]))
            res["splits"][sp] = {b: {"frames": len(v["iou"]), "alpha_soft_iou": round(float(np.mean(v["iou"])), 4), "rgb_err_in_mask_0_255": round(float(np.mean(v["rgb"])), 2),
                                     "token_swap_rgb_err": round(float(np.mean(v["swap_rgb"])), 2)} for b, v in sorted(acc.items())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res["splits"], indent=1))


if __name__ == "__main__":
    main()

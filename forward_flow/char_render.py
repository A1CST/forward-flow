"""Cartoon character renderer v1 (Claude 2026-10-17; Payton 1560847663044370454). Character lane of forward_flow_cartoon.

pose (13 joints, px) + character token (from ONE reference frame)  ->  figure-alone RGBA at 256 px.

- Pose features on a 128 grid: 13 joint heatmaps + 12 bone distance fields (the stick decoder's inputs, scaled to 256).
- Token: a small CNN reads the reference RGBA (the character's idle frame 0) once -> 256-d vector. At generation time the
  token is the only appearance input; target frames are never read (DATA_CONTRACT.md).
- U-Net with FiLM conditioning on the token at every level, 2x head to 256 px. Output: premultiplied RGB + alpha.

  python character/char_render.py train --pack character/packs/pilot_v1 --out character/runs/crender_v1 [--steps 20000]
  python character/char_render.py gate  --pack character/packs/pilot_v1 --model character/runs/crender_v1 --out .../gate.json
  python character/char_render.py calibrate --pack character/packs/pilot_v1 --out character/runs/calibration.json

Gate (held-out CHARACTERS, never trained on): soft alpha IoU and RGB error inside the true mask, against two controls:
  ref-paste  = the reference frame shifted so its pelvis sits on the target pelvis (right look, no pose change)
  token-swap = the model with another held-out character's token (right pose, wrong look)
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "render"))
S = 256; G = 128
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (8, 11), (11, 12))


def dev(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- pose features (128 grid, joints in 256 px)
def pose_features(J):
    J = J / (S / G)
    ys = torch.arange(G, device=J.device).float().view(1, 1, G, 1) + .5; xs = torch.arange(G, device=J.device).float().view(1, 1, 1, G) + .5
    hm = torch.exp(-((xs - J[..., 0, None, None]) ** 2 + (ys - J[..., 1, None, None]) ** 2) / (2 * 1.5 ** 2))
    b = torch.tensor(BONES, device=J.device); A, B = J[:, b[:, 0]], J[:, b[:, 1]]; ab = B - A; L2 = (ab ** 2).sum(-1).clamp_min(1e-6)
    ax, ay = A[..., 0, None, None], A[..., 1, None, None]
    t = (((xs - ax) * ab[..., 0, None, None] + (ys - ay) * ab[..., 1, None, None]) / L2[..., None, None]).clamp(0, 1)
    d2 = (xs - (ax + t * ab[..., 0, None, None])) ** 2 + (ys - (ay + t * ab[..., 1, None, None])) ** 2
    return torch.cat([hm, torch.exp(-d2 / 2), torch.exp(-d2 / 18)], 1)          # 13 + 12 sharp + 12 wide = 37


# ---------------------------------------------------------------- model
# ---------------------------------------------------------------- v2: palette read off the reference (Astra 1560858035956490301)
# Product-safe inputs only: the reference is a figure-alone cutout (its alpha = the observed segmentation) and its joints
# come from a pose reader, modelled in training AND gate as declared reader noise REF_JOINT_NOISE px. No renderer labels.
REF_JOINT_NOISE = 2.0
REGIONS = ((0,), (1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (8, 9), (9, 10), (8, 11), (11, 12), (4, 7), (10, 12))
#           head  torso   shoulders/arms (L,R)                        legs (L,R)                              hands  feet


def ref_palette(ref, Jr):
    """ref (N,4,256,256) straight RGBA, Jr (N,13,2) reader joints -> (N, 14*4) mean RGB + coverage per region."""
    N = len(ref); ys = torch.arange(S, device=ref.device).float().view(1, S, 1) + .5; xs = torch.arange(S, device=ref.device).float().view(1, 1, S) + .5
    a = ref[:, 3]; out = []
    for reg in REGIONS:
        if len(reg) == 1 or reg in ((4, 7), (10, 12)):
            d2 = torch.stack([(xs - Jr[:, j, 0, None, None]) ** 2 + (ys - Jr[:, j, 1, None, None]) ** 2 for j in reg]).min(0).values
            sig = 9.0 if reg == (0,) else 4.0
        else:
            A, B = Jr[:, reg[0]], Jr[:, reg[1]]; ab = B - A; L2 = (ab ** 2).sum(-1).clamp_min(1e-6)
            t = (((xs - A[:, 0, None, None]) * ab[:, 0, None, None] + (ys - A[:, 1, None, None]) * ab[:, 1, None, None]) / L2[:, None, None]).clamp(0, 1)
            d2 = (xs - (A[:, 0, None, None] + t * ab[:, 0, None, None])) ** 2 + (ys - (A[:, 1, None, None] + t * ab[:, 1, None, None])) ** 2
            sig = 10.0 if reg == (1, 8) else 3.0
        w = torch.exp(-d2 / (2 * sig ** 2)) * a; ws = w.sum((1, 2)).clamp_min(1e-4)
        out.append(torch.cat([(ref[:, :3] * w[:, None]).sum((2, 3)) / ws[:, None], (ws / (2 * math.pi * sig ** 2))[:, None].clamp(max=4) / 4], 1))
    return torch.cat(out, 1)


class TokenNet(nn.Module):
    def __init__(s, d=256):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.SiLU(), nn.Conv2d(o, o, 3, 1, 1), nn.SiLU())
        s.f = nn.Sequential(c(4, 32), c(32, 64), c(64, 128), c(128, 192), c(192, 256))   # 256 -> 8
        s.out = nn.Linear(256 * 2, d); s.v2 = False
    def forward(s, ref, Jr=None):                                                    # ref (N,4,256,256) straight RGBA in [0,1]
        h = s.f(ref); a = ref[:, 3:4]
        w = F.adaptive_avg_pool2d(a, h.shape[-1]); g = (h * w).sum((2, 3)) / w.sum((2, 3)).clamp_min(1e-4)   # figure-weighted pool
        z = s.out(torch.cat([g, h.mean((2, 3))], 1))
        if s.v2:
            z = z + s.pal(ref_palette(ref, Jr))
        return z


class Block(nn.Module):
    def __init__(s, i, o, d):
        super().__init__(); s.c1 = nn.Conv2d(i, o, 3, padding=1); s.c2 = nn.Conv2d(o, o, 3, padding=1); s.film = nn.Linear(d, 2 * o); s.n = nn.GroupNorm(8, o)
    def forward(s, x, z):
        h = F.silu(s.c1(x)); g, b = s.film(z).chunk(2, 1); h = s.n(h) * (1 + g[..., None, None]) + b[..., None, None]
        return F.silu(s.c2(h))


class CharRender(nn.Module):
    def __init__(s, d=256, c=(64, 128, 192, 256), v2=False, yaw_in=False, coords=False):
        super().__init__(); s.tok = TokenNet(d); s.v2 = v2
        s.coords = coords
        if v2:
            s.tok.v2 = True; s.tok.pal = nn.Sequential(nn.Linear(len(REGIONS) * 4, 256), nn.SiLU(), nn.Linear(256, d))
        s.yaw_in = yaw_in                                            # v3: declared facing yaw (sin, cos) as 2 input planes
        s.e = nn.ModuleList([Block(37 + (2 if yaw_in else 0), c[0], d), Block(c[0], c[1], d), Block(c[1], c[2], d), Block(c[2], c[3], d)])
        s.d = nn.ModuleList([Block(c[3] + c[2], c[2], d), Block(c[2] + c[1], c[1], d), Block(c[1] + c[0], c[0], d)])
        s.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(c[0], 48, 3, padding=1), nn.SiLU(),
                             nn.Conv2d(48, 48, 3, padding=1), nn.SiLU(), nn.Conv2d(48, 7 if s.coords else 4, 1))
    def forward(s, J, ref=None, z=None, Jr=None, yaw=None):
        z = s.tok(ref, Jr) if z is None else z; x = pose_features(J); hs = []
        if s.yaw_in:                                                  # front and back views share nearly the same 2D joints
            y = torch.zeros(len(J), device=J.device) if yaw is None else yaw
            x = torch.cat([x, torch.stack([torch.sin(y), torch.cos(y)], 1)[..., None, None].expand(-1, 2, *x.shape[-2:])], 1)
        for k, b in enumerate(s.e):
            x = b(x if k == 0 else F.avg_pool2d(x, 2), z); hs.append(x)
        for k, b in enumerate(s.d):
            x = b(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), hs[-2 - k]], 1), z)
        o = s.up(x); a = torch.sigmoid(o[:, 3:4]); rgb = torch.sigmoid(o[:, :3]) * a      # premultiplied
        if s.coords: s.last_coords = torch.sigmoid(o[:, 4:7])
        return torch.cat([rgb, a], 1)


# ---------------------------------------------------------------- data
def _read_u8(p): return np.asarray(Image.open(p).convert("RGBA"), np.uint8)


def load_rgba(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255).permute(2, 0, 1)


def premul(x): return torch.cat([x[:3] * x[3:4], x[3:4]], 0)


REF_POSE = "idle, phase 0, facing right, x 0 (rendered by make_pack.py as <char>/ref.png; not a frame of any clip)"
MIN_POSE_DIST = 0.12        # Astra 1560848806520819784: targets must be pose-disjoint from the reference (world units,
                            # mean joint distance after pelvis alignment and character-scale normalisation)


def ref_rel(scale):
    import cartoon_world as CW
    w = CW.pose("idle", 0.0, 0.0, 1); w[:, 1] *= scale; return w - w[8:9]


class Pack:
    def __init__(s, root, split, stride=1):
        s.root = Path(root); ix = json.loads((s.root / "index.json").read_text())["clips"]; s.items = []; s.yaws = []; s.refs = {}; s.pairs = []; s.excluded = 0
        for c in ix:
            if c["split"] != split: continue
            gt = json.loads((s.root / c["clip"] / "gt.json").read_text()); sc = gt["character"].get("scale", 1.0)
            rj = s.root / c["character"] / "ref.json"                       # gait packs declare their reference pose
            if rj.exists():
                w0 = np.asarray(json.loads(rj.read_text())["joints_world"], np.float32); rr = w0 - w0[8:9]
            else:
                rr = ref_rel(sc)
            if c["character"] not in s.refs:
                s.refs[c["character"]] = load_rgba(s.root / c["character"] / "ref.png")
            for f in gt["frames"][::stride]:
                w = np.asarray(f["joints_world"], np.float32); dist = float(np.linalg.norm((w - w[8:9]) - rr, axis=-1).mean() / sc)
                if dist < MIN_POSE_DIST:
                    s.excluded += 1; continue
                s.items.append((c["clip"], c["character"], f["t"], np.asarray(f["joints_px"], np.float32))); s.yaws.append(float(f.get("yaw", math.pi)) - math.pi)
                s.pairs.append({"reference": f"{c['character']}/ref.png", "target": f"{c['clip']}/rgba/{f['t']:05d}.png", "pose_dist": round(dist, 4)})
        s.chars = sorted(s.refs); s.cache = None
        reg = json.loads((s.root / "index.json").read_text()).get("regions")          # Blender packs: {region: grey level}
        s.regions = {k: (v,) for k, v in reg.items()} if reg else None
        import cartoon_world as CW
        sc = {c["character"]: json.loads((s.root / c["clip"] / "gt.json").read_text())["character"].get("scale", 1.0) for c in ix if c["split"] == split}
        def rj(ch):
            f = s.root / ch / "ref.json"
            if f.exists(): return torch.tensor(json.loads(f.read_text())["joints_px"], dtype=torch.float32)
            return torch.tensor(CW.project(CW.pose("idle", 0.0, 0.0, 1) * np.array([1, sc[ch]], np.float32), 0.0, S), dtype=torch.float32)
        s.ref_J = {ch: rj(ch) for ch in s.chars}
    def preload(s, d):
        """Payton 1560852577476673739: decode every target PNG ONCE (process pool) into one uint8 tensor on the device, so
        training steps only slice it (the per-step PIL decode left the GPU ~50% idle)."""
        from concurrent.futures import ProcessPoolExecutor
        import os
        paths = [str(s.root / c / "rgba" / f"{t:05d}.png") for c, _, t, _ in s.items]; t0 = time.time()
        with ProcessPoolExecutor(min(32, os.cpu_count() or 4)) as ex:
            arrs = list(ex.map(_read_u8, paths, chunksize=64))
        s.cache = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous().to(d)
        s.ref_t = torch.stack([s.refs[c] for c in s.chars]).to(d); s.char_ix = torch.tensor([s.chars.index(it[1]) for it in s.items], device=d)
        s.J_t = torch.tensor(np.stack([it[3] for it in s.items]), device=d); s.refJ_t = torch.stack([s.ref_J[c] for c in s.chars]).to(d)
        print(f"preloaded {len(paths)} targets ({s.cache.numel() / 1e9:.2f} GB uint8) in {time.time() - t0:.0f}s", flush=True)
    def manifest(s, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"reference_pose": REF_POSE, "min_pose_dist": MIN_POSE_DIST, "excluded_near_reference": s.excluded,
                                          "n_pairs": len(s.pairs), "min_pair_dist": min(p["pose_dist"] for p in s.pairs), "pairs": s.pairs}))
    def yaw(s, idx, d):
        return torch.tensor([s.yaws[i] for i in (idx.tolist() if hasattr(idx, "tolist") else idx)], dtype=torch.float32, device=d)

    def batch(s, idx, d):
        if s.cache is not None:
            ix = torch.as_tensor(idx, device=d); x = s.cache[ix].float() / 255
            return s.J_t[ix], s.ref_t[s.char_ix[ix]], torch.cat([x[:, :3] * x[:, 3:4], x[:, 3:4]], 1), s.refJ_t[s.char_ix[ix]]
        J = torch.tensor(np.stack([s.items[i][3] for i in idx]), device=d)
        R = torch.stack([s.refs[s.items[i][1]] for i in idx]).to(d)
        T = torch.stack([premul(load_rgba(s.root / s.items[i][0] / "rgba" / f"{s.items[i][2]:05d}.png")) for i in idx]).to(d)
        return J, R, T, torch.stack([s.ref_J[s.items[i][1]] for i in idx]).to(d)


def colour_aug(R, T, g):
    """v2: recolour reference and target TOGETHER (channel shuffle + per-channel gain), so palettes cannot be memorised."""
    n = len(R); perm = torch.stack([torch.randperm(3, generator=g) for _ in range(n)]).to(R.device)
    gain = (0.55 + 0.9 * torch.rand(n, 3, generator=g)).to(R.device)
    def f(rgb, a):
        x = torch.gather(rgb, 1, perm[:, :, None, None].expand_as(rgb)) * gain[:, :, None, None]
        return torch.minimum(x, a)                                   # premultiplied / straight-with-alpha stays valid
    return torch.cat([f(R[:, :3], torch.ones_like(R[:, 3:4])), R[:, 3:4]], 1), torch.cat([f(T[:, :3], T[:, 3:4]), T[:, 3:4]], 1)


def loss_fn(P, T):
    a, ta = P[:, 3:4], T[:, 3:4]; w = 1 + 4 * (ta > 0).float()
    return (w * F.binary_cross_entropy(a.clamp(1e-5, 1 - 1e-5), ta, reduction="none")).mean() + 4 * (w * (P[:, :3] - T[:, :3]).abs()).mean()


def train(a):
    torch.manual_seed(20261017); d = dev(); P = Pack(a.pack, "train"); net = CharRender(v2=a.v2, yaw_in=a.yaw).to(d); g = torch.Generator().manual_seed(7)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); P.manifest(out / "pairs_train.json")
    if not os.environ.get("CR_NO_PRELOAD"): P.preload(d)          # CR_NO_PRELOAD=1: desktop smoke tests only
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=.05)
    rng = np.random.default_rng(0); t0 = time.time(); hist = []
    print(f"train frames={len(P.items)} chars={len(P.chars)} params={sum(p.numel() for p in net.parameters())}", flush=True)
    for st in range(a.steps):
        bi = rng.integers(len(P.items), size=a.bs); J, R, T, Jr = P.batch(bi, d); Y = P.yaw(bi, d)
        J = J + torch.randn_like(J) * .3                                             # pose-reading noise robustness
        if a.v2:
            R, T = colour_aug(R, T, g); Jr = Jr + torch.randn_like(Jr) * REF_JOINT_NOISE  # declared reference-reader noise
        l = loss_fn(net(J, R, Jr=Jr, yaw=Y), T); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        hist.append(float(l))
        if st % 500 == 0 or st == a.steps - 1:
            print(f"step={st} loss={np.mean(hist[-500:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if a.plateau and st >= 6000 and st % 1000 == 0 and np.mean(hist[-1000:]) > .99 * np.mean(hist[-2000:-1000]):
            print(f"plateau stop at step {st}: last-1000 mean {np.mean(hist[-1000:]):.4f} vs previous {np.mean(hist[-2000:-1000]):.4f}", flush=True); break
    torch.save(net.cpu().state_dict(), out / "char_render.pt")
    (out / "train.json").write_text(json.dumps({"pack": str(a.pack), "steps": a.steps, "steps_run": st + 1, "plateau_rule": "stop when the last 1000-step mean loss is not 1% below the previous 1000 (from step 6000)" if a.plateau else None, "bs": a.bs, "v2": a.v2, "yaw_in": a.yaw, "ref_joint_noise_px": REF_JOINT_NOISE if a.v2 else None, "colour_aug": "channel shuffle + gain .55-1.45 on reference and target together" if a.v2 else None, "train_chars": len(P.chars), "frames": len(P.items),
                                                "params": sum(p.numel() for p in net.parameters()), "final_loss_500": float(np.mean(hist[-500:]))}))


# ---------------------------------------------------------------- metrics
def soft_iou(pa, ta): return float(torch.minimum(pa, ta).sum() / torch.maximum(pa, ta).sum().clamp_min(1e-6))


def rgb_err(P, T):                                     # mean |rgb| in 0-255 inside the true figure (straight colour)
    m = T[3] > .5
    if m.sum() == 0: return 0.0
    pc = P[:3] / P[3:4].clamp_min(1e-3); tc = T[:3] / T[3:4].clamp_min(1e-3)
    return float((pc - tc).abs()[:, m].mean() * 255)


def shift(img, dx, dy):                                # integer translate (N,C,H,W) with zero fill
    out = torch.zeros_like(img); H, W = img.shape[-2:]; dx, dy = int(round(dx)), int(round(dy))
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd] = img[..., ys, xs]; return out


@torch.no_grad()
def gate(a):
    d = dev(); net = load_model(a.model).to(d)
    P = Pack(a.pack, "heldout", stride=4); res = {"heldout_chars": len(P.chars), "frames": len(P.items), "excluded_near_reference": P.excluded}
    P.manifest(Path(a.out).parent / "pairs_gate.json")
    import cartoon_world as CW
    gn = torch.Generator().manual_seed(11)                                            # same declared reader noise as training
    toks = {c: net.tok(P.refs[c][None].to(d), (P.ref_J[c] + torch.randn(13, 2, generator=gn) * REF_JOINT_NOISE)[None].to(d) if net.v2 else None) for c in P.chars}
    m = {"model": [[], []], "ref_paste": [[], []], "token_swap": [[], []]}; reg = {"model": {}, "token_swap": {}}
    for i in range(0, len(P.items), 32):
        idx = list(range(i, min(i + 32, len(P.items)))); J, R, T, _ = P.batch(idx, d)
        Y = P.yaw(idx, d); out = net(J, z=torch.cat([toks[P.items[k][1]] for k in idx]), yaw=Y)
        sw = net(J, z=torch.cat([toks[P.chars[(P.chars.index(P.items[k][1]) + 1) % len(P.chars)]] for k in idx]), yaw=Y)
        for j, k in enumerate(idx):
            ch = P.items[k][1]; refJ = P.ref_J[ch].numpy()
            rp = shift(premul(P.refs[ch])[None].to(d), *(P.items[k][3][8] - refJ[8]))[0]
            for name, X in (("model", out[j]), ("ref_paste", rp), ("token_swap", sw[j])):
                m[name][0].append(soft_iou(X[3], T[j][3])); m[name][1].append(rgb_err(X, T[j]))
            lab = torch.from_numpy(np.asarray(Image.open(P.root / P.items[k][0] / "labels" / f"{P.items[k][2]:05d}.png").convert("L")).astype(np.int16)).to(d)   # gate-only truth
            for name, X in (("model", out[j]), ("token_swap", sw[j])):
                for rn, ids in (P.regions or REGION_LABELS).items():
                    e = region_err(X, T[j], lab, ids)
                    if e is not None: reg[name].setdefault(rn, []).append(e)
    for name, (iou, err) in m.items():
        res[name] = {"alpha_soft_iou": round(float(np.mean(iou)), 4), "alpha_iou_p5": round(float(np.percentile(iou, 5)), 4), "rgb_err_in_mask_0_255": round(float(np.mean(err)), 2)}
    for name, r in reg.items():
        res[name]["region_rgb_err_0_255"] = {rn: round(float(np.mean(v)), 2) for rn, v in r.items()}
    res["region_note"] = "Astra 1560858035956490301: per-region colour error (renderer part labels, gate only) so one wrong region cannot hide in the mean"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


REGION_LABELS = {"skin_head": (1,), "shirt_torso_arms": (2, 3, 4, 5, 6), "pants_legs": (7, 8, 9, 10), "accent_hands_feet": (11, 12)}


def region_err(P, T, lab, ids):
    m = torch.zeros_like(lab, dtype=torch.bool)
    for i in ids: m |= (lab - i).abs() <= 3                                    # Blender label pass: grey levels +-1
    m &= T[3] > .5
    if m.sum() < 8: return None
    pc = P[:3] / P[3:4].clamp_min(1e-3); tc = T[:3] / T[3:4].clamp_min(1e-3)
    return float((pc - tc).abs()[:, m].mean() * 255)


def load_model(path):
    tj = json.loads((Path(path) / "train.json").read_text()); net = CharRender(v2=bool(tj.get("v2")), yaw_in=bool(tj.get("yaw_in")), coords=bool(tj.get("coords")))
    net.load_state_dict(torch.load(Path(path) / "char_render.pt", map_location="cpu")); return net.eval()


def calibrate_shift(a):
    """Blender packs (no procedural re-draw with jittered joints): the metric's ceiling for a 1 px misregistration =
    each held-out truth frame vs itself shifted by 1 px in x and in y (mean over both)."""
    P = Pack(a.pack, "heldout", stride=8); iou, err = [], []
    for clip, ch, t, J in P.items[:600]:
        T = premul(load_rgba(P.root / clip / "rgba" / f"{t:05d}.png"))
        for dx, dy in ((1, 0), (0, 1)):
            X = shift(T[None], dx, dy)[0]; iou.append(soft_iou(X[3], T[3])); err.append(rgb_err(X, T))
    res = {"shift_1px": {"alpha_soft_iou": round(float(np.mean(iou)), 4), "rgb_err_in_mask_0_255": round(float(np.mean(err)), 2)}, "frames": len(P.items[:600])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def calibrate(a):
    """Truth-vs-truth ceiling of the metrics: the renderer's own figure with joints jittered by 0.25 / 0.5 / 1 px."""
    import cartoon_world as CW
    P = Pack(a.pack, "heldout", stride=8); rng = np.random.default_rng(0); res = {}
    for jit in (.25, .5, 1.0):
        iou, err = [], []
        for clip, ch, t, J in P.items[:400]:
            g = json.loads((P.root / clip / "gt.json").read_text()); C = CW.Character(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in g["character"].items()})
            T = premul(load_rgba(P.root / clip / "rgba" / f"{t:05d}.png"))
            X = premul(torch.from_numpy(np.asarray(CW.draw_character(S, J + rng.normal(0, jit, J.shape), C)[0].convert("RGBA"), np.float32) / 255).permute(2, 0, 1))
            iou.append(soft_iou(X[3], T[3])); err.append(rgb_err(X, T))
        res[f"jitter_{jit}px"] = {"alpha_soft_iou": round(float(np.mean(iou)), 4), "rgb_err_in_mask_0_255": round(float(np.mean(err)), 2)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("train", "gate", "calibrate", "calibrate-shift")); ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--model"); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--plateau", action="store_true"); ap.add_argument("--v2", action="store_true"); ap.add_argument("--yaw", action="store_true")
    a = ap.parse_args(); {"train": train, "gate": gate, "calibrate": calibrate, "calibrate-shift": calibrate_shift}[a.cmd](a)


if __name__ == "__main__":
    main()

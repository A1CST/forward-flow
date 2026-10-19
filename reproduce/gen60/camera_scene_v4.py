"""v4: PERSPECTIVE street for GEN60 (handoffs/scene_entity_texture/gen60/GEN60_CARD.md), same layer map as v3
(LAYER_MAP_v1_APPROVED.md): L0 scene (sky, ground, registry) -> L1 camera (pinhole, observes only) -> L1 buildings (no model;
declared from a seeded generator) -> L2 children in BUILDING-LOCAL facade coordinates (door, windows, awning, sign, roof trim).
Culling tests each parent's SUBTREE bound (parent volume + all children) against the viewport plus a guard band.
Procedural renderer, not a learned model. v1-v3 are untouched.
v4.1 (Payton 1561554807041425430 glitches): painter order by nearest-point distance; v4.2: exact separating-axis order.
World: X along the street, Y up (m), Z across (centre 0). Facades face the street at z = +-FACADE_Z.
"""
from __future__ import annotations
import math, random
from PIL import Image, ImageDraw

W = H = 512
F_PX = 430.0                                  # focal length in px (~62 deg vertical FOV at 512)
NEAR = 0.15
FACADE_Z = 13.5
INK = (45, 43, 53)
SKY, ROAD, WALK, CURB, LANE = (150, 200, 235), (112, 116, 126), (190, 190, 186), (160, 160, 158), (236, 228, 196)
PALETTE = [(220, 190, 91), (237, 170, 91), (203, 150, 190), (221, 188, 90), (160, 196, 160), (196, 120, 104), (150, 170, 210),
           (230, 214, 190), (180, 150, 120), (120, 160, 170), (240, 200, 160), (170, 130, 160)]
DOOR_C, WIN_C, FRAME_C, AWN_C, SIGN_C = (96, 66, 52), (176, 214, 236), (45, 43, 53), (200, 70, 70), (250, 240, 210)
EDIT, UNCULL_ALL = {}, False


def shade(c, f): return tuple(max(0, min(255, round(v * f))) for v in c)


def make_street(seed=20261018, x0=-25.0, x1=110.0):
    """Seeded building generator: both sides, varied width / depth / height / roof / facade / window grid / awning / sign."""
    rng = random.Random(seed); world = []
    for side in (-1, 1):
        x = x0 + rng.uniform(0, 4)
        while x < x1:
            w = rng.uniform(5.0, 13.0); h = rng.choice([rng.uniform(6, 10), rng.uniform(10, 18), rng.uniform(18, 30)])
            d = rng.uniform(6, 12); setback = rng.choice([0.0, 0.0, 0.6, 1.4]); roof = rng.choice(["flat", "flat", "step", "gable"])
            b = {"id": f"bld_{'S' if side < 0 else 'N'}{len([o for o in world if o['side'] == side]):02d}", "side": side,
                 "x0": x, "x1": x + w, "zf": side * (FACADE_Z + setback), "d": d, "h": h, "roof": roof,
                 "colour": rng.choice(PALETTE), "win_rows": max(1, int((h - 3.2) // rng.uniform(2.6, 3.6))),
                 "win_cols": max(1, int(w // rng.uniform(1.8, 3.0))), "win_style": rng.choice(["square", "tall", "wide"]),
                 "door_u": rng.uniform(0.15, 0.85), "awning": rng.random() < 0.45, "sign": rng.random() < 0.35}
            b["children"] = children_of(b); world.append(b)
            x += w + rng.choice([0.0, 0.0, 0.3, 1.5])
    return world


def children_of(b):
    """L2 children in facade-local metres (u along the facade from x0, v up from the ground)."""
    w, h = b["x1"] - b["x0"], b["h"]; out = []
    du = 1.2; out.append({"id": f"{b['id']}/door", "kind": "door", "u": max(0.2, min(w - du - 0.2, b["door_u"] * w - du / 2)), "v": 0.0, "du": du, "dv": 2.2})
    if b["awning"]: out.append({"id": f"{b['id']}/awning", "kind": "awning", "u": 0.3, "v": 2.6, "du": w - 0.6, "dv": 0.5})
    if b["sign"]: out.append({"id": f"{b['id']}/sign", "kind": "sign", "u": w * 0.25, "v": 3.3, "du": w * 0.5, "dv": 0.8})
    ww, wh = {"square": (1.0, 1.0), "tall": (0.8, 1.6), "wide": (1.6, 0.9)}[b["win_style"]]
    cols, rows = b["win_cols"], b["win_rows"]
    for r in range(rows):
        v = 4.4 + r * (h - 5.0) / max(1, rows)
        if v + wh > h - 0.6: break
        for c in range(cols):
            u = (c + 0.5) * w / cols - ww / 2
            out.append({"id": f"{b['id']}/win_r{r}c{c}", "kind": "window", "u": u, "v": v, "du": ww, "dv": wh})
    if b["roof"] != "gable": out.append({"id": f"{b['id']}/trim", "kind": "trim", "u": 0.0, "v": h - 0.35, "du": w, "dv": 0.35})
    return out


class Cam:
    def __init__(s, C, L, f=F_PX):
        s.C = C; fx, fy, fz = (L[0] - C[0], L[1] - C[1], L[2] - C[2]); n = math.sqrt(fx * fx + fy * fy + fz * fz)
        s.fwd = (fx / n, fy / n, fz / n); rx, rz = -s.fwd[2], s.fwd[0]; rn = math.hypot(rx, rz) or 1.0
        s.right = (rx / rn, 0.0, rz / rn); fw = s.fwd
        s.up = (s.right[1] * fw[2] - s.right[2] * fw[1], s.right[2] * fw[0] - s.right[0] * fw[2], s.right[0] * fw[1] - s.right[1] * fw[0])
        if s.up[1] < 0: s.right = tuple(-v for v in s.right); s.up = tuple(-v for v in s.up)
        s.f = f

    def cam(s, p):
        d = (p[0] - s.C[0], p[1] - s.C[1], p[2] - s.C[2])
        return (sum(a * b for a, b in zip(d, s.right)), sum(a * b for a, b in zip(d, s.up)), sum(a * b for a, b in zip(d, s.fwd)))

    def px(s, q): return (W / 2 + s.f * q[0] / q[2], H / 2 - s.f * q[1] / q[2])


def clip_near(poly):
    """Sutherland-Hodgman against z = NEAR in camera space."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]; ia, ib = a[2] >= NEAR, b[2] >= NEAR
        if ia: out.append(a)
        if ia != ib:
            t = (NEAR - a[2]) / (b[2] - a[2]); out.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))
    return out


def face(cam, pts):
    q = clip_near([cam.cam(p) for p in pts])
    return [cam.px(v) for v in q] if len(q) >= 3 else None


def facade_pt(b, u, v, off=0.0):
    return (b["x0"] + u, v, b["zf"] - b["side"] * off)       # off > 0 = towards the street (awnings, signs)


def bounds(b):
    zf, zb = b["zf"], b["zf"] + b["side"] * b["d"]; top = b["h"] + (2.5 if b["roof"] == "gable" else 0) + (3 if b["roof"] == "step" else 0)
    return [(x, y, z) for x in (b["x0"], b["x1"]) for y in (0, top) for z in (zf - b["side"] * 0.8, zb)]


def aabb(b):
    zs = (b["zf"] - b["side"] * 0.9, b["zf"] + b["side"] * b["d"]); top = b["h"] + (2.5 if b["roof"] == "gable" else 0) + (3 if b["roof"] == "step" else 0)
    return (b["x0"], 0.0, min(zs)), (b["x1"], top, max(zs))


def nearest_dist(b, C):
    lo, hi = aabb(b); return math.dist(C, tuple(min(max(C[k], lo[k]), hi[k]) for k in range(3)))


def after(A, B, C, eps=1e-6):
    """+1: box A must be drawn after (in front of) box B; -1: the reverse; 0: no separating axis (overlap) -> distance fallback.
    Disjoint AABBs are separated along some axis k; the box on the camera's side of the gap is in front. A camera inside the gap
    sees them on opposite sides (no overlap on screen), so either order is correct there."""
    (alo, ahi), (blo, bhi) = A, B
    for k in range(3):
        if ahi[k] <= blo[k] + eps: return 1 if C[k] <= (ahi[k] + blo[k]) / 2 else -1
        if bhi[k] <= alo[k] + eps: return -1 if C[k] <= (bhi[k] + alo[k]) / 2 else 1
    return 0


def draw_order(items, C):
    """Topological painter order (far first) from pairwise separating-axis tests; ties / cycles broken by nearest-point distance."""
    n = len(items); dist = [math.dist(C, tuple(min(max(C[k], it[2][0][k]), it[2][1][k]) for k in range(3))) for it in items]
    succ = [set() for _ in range(n)]; indeg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            r = after(items[i][2], items[j][2], C) or (1 if dist[i] < dist[j] else -1)
            a, b = (j, i) if r > 0 else (i, j)                                  # edge a -> b: a drawn before b
            if b not in succ[a]: succ[a].add(b); indeg[b] += 1
    out, ready, done = [], [i for i in range(n) if indeg[i] == 0], set()
    while len(out) < n:
        if not ready: ready = [max((i for i in range(n) if i not in done), key=lambda i: dist[i])]   # cycle: farthest first
        i = max(ready, key=lambda i: dist[i]); ready.remove(i)
        if i in done: continue
        done.add(i); out.append(items[i])
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0 and j not in done: ready.append(j)
    return out


def visible(cam, b, guard=24):
    pts = [cam.cam(p) for p in bounds(b)]
    if all(p[2] < NEAR for p in pts): return False
    ps = [cam.px(p) for p in pts if p[2] >= NEAR]
    if len(ps) < len(pts): return True                       # straddles the camera plane: keep (force-render rule)
    xs, ys = [p[0] for p in ps], [p[1] for p in ps]
    return not (max(xs) < -guard or min(xs) > W + guard or max(ys) < -guard or min(ys) > H + guard)


def box_faces(b, cam):
    """Facade, the two side walls, roof; returns (depth, polygon, colour) for painter order; back faces culled."""
    x0, x1, zf, zb, h, s = b["x0"], b["x1"], b["zf"], b["zf"] + b["side"] * b["d"], b["h"], b["side"]
    quads = {"facade": ([(x0, 0, zf), (x1, 0, zf), (x1, h, zf), (x0, h, zf)], (0, 0, -s), 1.0),
             "left": ([(x0, 0, zb), (x0, 0, zf), (x0, h, zf), (x0, h, zb)], (-1, 0, 0), .72),
             "right": ([(x1, 0, zf), (x1, 0, zb), (x1, h, zb), (x1, h, zf)], (1, 0, 0), .78),
             "roof": ([(x0, h, zf), (x1, h, zf), (x1, h, zb), (x0, h, zb)], (0, 1, 0), .62)}
    out = []
    for name, (pts, n, f) in quads.items():
        cx, cy, cz = (sum(p[k] for p in pts) / 4 for k in range(3))
        if (cam.C[0] - cx) * n[0] + (cam.C[1] - cy) * n[1] + (cam.C[2] - cz) * n[2] <= 0: continue
        poly = face(cam, pts)
        if poly: out.append((name, poly, shade(b["colour"], f)))
    if b["roof"] == "gable":
        for z in (zf, zb):
            tri = face(cam, [(x0, h, z), (x1, h, z), ((x0 + x1) / 2, h + 2.5, z)])
            if tri: out.append(("gable", tri, shade(b["colour"], .9)))
    if b["roof"] == "step":
        inset = (x1 - x0) * 0.2; sb = dict(b, x0=x0 + inset, x1=x1 - inset, zf=zf + s * 1.5, d=b["d"] - 2, h=h + 3, roof="flat")
        st = [(f"step_{n_}", p, c) for n_, p, c in box_faces(sb, cam)]
        out = (st + out) if cam.C[1] < h else (out + st)
    return out


def draw_children(draw, b, cam):
    if (cam.C[2] - b["zf"]) * -b["side"] <= 0: return {}       # facade faces away from the camera: children hidden with it
    rects = {}
    for ch in b["children"]:
        off = {"awning": 0.9, "sign": 0.15}.get(ch["kind"], 0.02)
        poly = face(cam, [facade_pt(b, ch["u"], ch["v"], off), facade_pt(b, ch["u"] + ch["du"], ch["v"], off),
                          facade_pt(b, ch["u"] + ch["du"], ch["v"] + ch["dv"], off), facade_pt(b, ch["u"], ch["v"] + ch["dv"], off)])
        if not poly: continue
        col = EDIT.get(ch["id"], {"door": DOOR_C, "window": WIN_C, "awning": AWN_C, "sign": SIGN_C, "trim": shade(b["colour"], .55)}[ch["kind"]])
        draw.polygon(poly, fill=col, outline=FRAME_C); rects[ch["id"]] = [round(v, 3) for xy in poly for v in xy]
    return rects


def ground(draw, cam, x_lo, x_hi):
    def strip(z0, z1, col):
        p = face(cam, [(x_lo, 0, z0), (x_hi, 0, z0), (x_hi, 0, z1), (x_lo, 0, z1)])
        if p: draw.polygon(p, fill=col)
    strip(-FACADE_Z - 2, -4.7, WALK); strip(4.7, FACADE_Z + 2, WALK); strip(-4.7, -4.5, CURB); strip(4.5, 4.7, CURB); strip(-4.5, 4.5, ROAD)
    for k in range(int(x_lo // 6) - 1, int(x_hi // 6) + 2):
        p = face(cam, [(k * 6.0, 0.005, -0.12), (k * 6.0 + 3.0, 0.005, -0.12), (k * 6.0 + 3.0, 0.005, 0.12), (k * 6.0, 0.005, 0.12)])
        if p: draw.polygon(p, fill=LANE)


def render(world, cam, character=None, char_ground=None, char_height_m=1.75, joints=None):
    """character: RGBA sprite (figure cut from the 256 canvas); char_ground: his ground point (x, 0, z).
    Draw order: sky, ground, buildings far -> near, the character inserted at his depth."""
    img = Image.new("RGB", (W, H), SKY); draw = ImageDraw.Draw(img)
    ground(draw, cam, cam.C[0] - 120, cam.C[0] + 120)
    vis = [b for b in world if UNCULL_ALL or visible(cam, b)]
    items = [("b", b, aabb(b)) for b in vis]                                   # v4.2: exact order for disjoint boxes (separating axis)
    if character is not None and char_ground is not None:
        items.append(("c", None, ((char_ground[0] - .35, 0.0, char_ground[2] - .35), (char_ground[0] + .35, char_height_m, char_ground[2] + .35))))
    items = draw_order(items, cam.C)
    rects, drawn = {}, []
    for kind, b, _ in items:
        if kind == "b":
            for name, poly, col in box_faces(b, cam):                      # one convex box: visible faces never overlap
                draw.polygon(poly, fill=col, outline=INK)
            rects.update(draw_children(draw, b, cam)); drawn.append(b["id"])
        else:
            g, hd = cam.cam(char_ground), cam.cam((char_ground[0], char_height_m, char_ground[2]))
            if g[2] < NEAR: continue
            (gx, gy), (_, hy) = cam.px(g), cam.px(hd); hpx = max(4, gy - hy)
            if joints is not None:                                             # v4.3: place the whole 256 canvas from the rig joints (same for any sprite)
                J = joints; ay = max(J[10][1], J[12][1]); ax = (J[10][0] + J[12][0]) / 2; hj = cam.px(cam.cam((char_ground[0], 1.62, char_ground[2])))[1]
                sc = max(1e-3, (gy - hj) / max(1.0, ay - J[0][1])); n = max(1, round(character.width * sc))
                sp = character.resize((n, n), Image.Resampling.LANCZOS); img.paste(sp, (round(gx - ax * sc), round(gy - ay * sc)), sp)
            else:
                bb = character.getchannel("A").getbbox(); fig = character.crop(bb) if bb else character
                wpx = max(1, round(fig.width * hpx / fig.height)); sp = fig.resize((wpx, max(1, round(hpx))), Image.Resampling.LANCZOS)
                img.paste(sp, (round(gx - wpx / 2), round(gy - hpx)), sp)
            drawn.append("character")
    render.last_rects = rects
    return img, drawn
